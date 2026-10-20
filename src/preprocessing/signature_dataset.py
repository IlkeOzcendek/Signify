import hashlib
import os
import random
import re
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from preprocess import preprocess_signature

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff"
}


def _preprocess_version():
    # preprocess.py'nin içerik hash'i. Dosya değişirse hash değişir, cache
    # klasörü değişir ve eski (bayat) cache otomatik geçersiz olur.
    try:
        import preprocess
        source = Path(preprocess.__file__).read_bytes()
        return hashlib.md5(source).hexdigest()[:10]
    except Exception:
        return "nover"


# Preprocess çıktısı deterministiktir; her epoch yeniden hesaplamak yerine
# bir kez diske yazıp tekrar okuruz. Augmentation cache'lenmez (rastgele).
# Training, validation and application verification must use exactly the same
# preprocessing route.  The explicit pipeline tag also invalidates caches made
# by the former UTSig-only preprocessing branch.
PREPROCESS_PIPELINE = "unified_app_v1"
PREPROCESS_CACHE_DIR = (
    Path(".cache") / f"preprocess_{PREPROCESS_PIPELINE}_{_preprocess_version()}"
)

def list_images(folder):
    if not folder.exists():
        return []

    return [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def extract_first_number(text):
    match = re.search(r"\d+", text)

    if match is None:
        return None

    return int(match.group())


def extract_utsig_person_id(file_path, dataset_root):
    relative_path = file_path.relative_to(dataset_root)

    # Önce klasör adlarından kişi numarası bulmaya çalışır.
    for part in relative_path.parts[:-1]:
        person_number = extract_first_number(part)

        if person_number is not None:
            return person_number

    # Klasörlerden bulunamazsa dosya adının ilk sayısını kullanır.
    return extract_first_number(file_path.stem)


def discover_person_ids():
    person_ids = set()

    # CEDAR: 1–55
    cedar_org_dir = Path("dataset/cedar/signatures/full_org")

    for file_path in list_images(cedar_org_dir):
        parts = file_path.stem.split("_")

        if len(parts) > 1 and parts[1].isdigit():
            person_ids.add(int(parts[1]))

    # Custom selected: 1001+
    custom_dir = Path("dataset/custom_selected")

    if custom_dir.exists():
        for person_dir in custom_dir.iterdir():
            if not person_dir.is_dir():
                continue

            if person_dir.name.endswith("_forg"):
                continue

            if person_dir.name.isdigit():
                person_ids.add(1000 + int(person_dir.name))

    # UTSig: 2001+
    utsig_genuine_dir = Path("dataset/UTSig/Genuine")

    for file_path in list_images(utsig_genuine_dir):
        base_id = extract_utsig_person_id(
            file_path,
            utsig_genuine_dir
        )

        if base_id is not None:
            person_ids.add(2000 + base_id)

    return sorted(person_ids)


def split_person_ids(seed=42, train_frac=0.70, val_frac=0.15):
    # Kişi kimliklerini keşfet + deterministik karıştır + train/val/test'e böl.
    # train.py, evaluate.py ve test.py bu tek fonksiyonu kullanır; böylece
    # bölme her yerde AYNI olur ve test seti gerçekten held-out kalır.
    all_person_ids = discover_person_ids()

    if len(all_person_ids) == 0:
        raise ValueError("No people were found in the configured datasets.")

    generator = torch.Generator().manual_seed(seed)

    shuffled_indexes = torch.randperm(
        len(all_person_ids),
        generator=generator
    ).tolist()

    all_person_ids = [all_person_ids[index] for index in shuffled_indexes]

    total_people = len(all_person_ids)
    train_end = int(total_people * train_frac)
    val_end = train_end + int(total_people * val_frac)

    train_ids = all_person_ids[:train_end]
    val_ids = all_person_ids[train_end:val_end]
    test_ids = all_person_ids[val_end:]

    return train_ids, val_ids, test_ids


class SignatureDataset(Dataset):

    def __init__(self, person_ids=None, augment=False, seed=42,
                 max_positive_pairs_per_person=None):
        self.cedar_dir = Path("dataset/cedar/signatures")
        self.custom_dir = Path("dataset/custom_selected")
        self.utsig_dir = Path("dataset/UTSig")

        self.person_ids = set(person_ids) if person_ids is not None else None
        self.augment = augment
        self.random = random.Random(seed)

        # Kişi başına pozitif pair üst sınırı (None = sınırsız). Negatifler
        # pozitife göre dengelendiği için bu, toplam pair sayısını da kısar.
        self.max_positive_pairs_per_person = max_positive_pairs_per_person

        self.base_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # Daha güçlü augmentation: overfit'i azaltır ve gerçek-dünya
        # imzalarındaki çeşitliliğe (farklı kalem/açı/basınç) dayanıklılık verir.
        self.augment_transform = transforms.Compose([
            transforms.RandomAffine(
                degrees=12,   # ±12° eğiklik toleransı (hafif eğik imzalar için)
                translate=(0.08, 0.08),
                scale=(0.85, 1.15),
                shear=8,
                fill=255
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.25)
        ])

        self.original_records = []
        self.forged_records = []

        self.load_cedar_records()
        self.load_custom_records()
        self.load_utsig_records()

        self.pairs = []
        self.build_pairs()

    def add_record(self, record_list, file_path, person_id):
        if self.person_ids is None or person_id in self.person_ids:
            record_list.append((file_path, person_id))

    def load_cedar_records(self):
        org_dir = self.cedar_dir / "full_org"
        forg_dir = self.cedar_dir / "full_forg"

        for file_path in list_images(org_dir):
            parts = file_path.stem.split("_")

            if len(parts) > 1 and parts[1].isdigit():
                person_id = int(parts[1])

                self.add_record(
                    self.original_records,
                    file_path,
                    person_id
                )

        for file_path in list_images(forg_dir):
            parts = file_path.stem.split("_")

            if len(parts) > 1 and parts[1].isdigit():
                person_id = int(parts[1])

                self.add_record(
                    self.forged_records,
                    file_path,
                    person_id
                )

    def load_custom_records(self):
        if not self.custom_dir.exists():
            return

        for person_dir in self.custom_dir.iterdir():
            if not person_dir.is_dir():
                continue

            folder_name = person_dir.name
            is_forgery = folder_name.endswith("_forg")

            numeric_name = folder_name.replace("_forg", "")

            if not numeric_name.isdigit():
                continue

            person_id = 1000 + int(numeric_name)

            destination = (
                self.forged_records
                if is_forgery
                else self.original_records
            )

            for file_path in list_images(person_dir):
                self.add_record(
                    destination,
                    file_path,
                    person_id
                )

    def load_utsig_records(self):
        genuine_dir = self.utsig_dir / "Genuine"
        forgery_dir = self.utsig_dir / "Forgery"

        # Genuine
        for file_path in list_images(genuine_dir):
            base_id = extract_utsig_person_id(
                file_path,
                genuine_dir
            )

            if base_id is None:
                continue

            person_id = 2000 + base_id

            self.add_record(
                self.original_records,
                file_path,
                person_id
            )

        # Opposite Hand + Simple + Skilled hepsi dahil edilir.
        for file_path in list_images(forgery_dir):
            base_id = extract_utsig_person_id(
                file_path,
                forgery_dir
            )

            if base_id is None:
                continue

            person_id = 2000 + base_id

            self.add_record(
                self.forged_records,
                file_path,
                person_id
            )

    def group_by_person(self, records):
        grouped = defaultdict(list)

        for file_path, person_id in records:
            grouped[person_id].append(file_path)

        return grouped

    def build_pairs(self):
        originals = self.group_by_person(self.original_records)
        forgeries = self.group_by_person(self.forged_records)

        person_ids = sorted(originals.keys())

        for person_id in person_ids:
            original_list = originals[person_id]

            if len(original_list) < 2:
                continue

            positive_pairs = []

            for i in range(len(original_list)):
                for j in range(i + 1, len(original_list)):
                    positive_pairs.append(
                        (original_list[i], original_list[j], 1)
                    )

            # Sınır varsa pozitif pair'leri örnekleyerek kırp (hızlı baseline).
            if (self.max_positive_pairs_per_person is not None
                    and len(positive_pairs) > self.max_positive_pairs_per_person):
                positive_pairs = self.random.sample(
                    positive_pairs,
                    self.max_positive_pairs_per_person
                )

            negative_candidates = []

            # Aynı kişiyi taklit eden sahte imzalar
            for forged in forgeries.get(person_id, []):
                for original in original_list:
                    negative_candidates.append(
                        (original, forged, 0)
                    )

            # Başka kişilerin genuine imzaları
            other_person_ids = [
                other_id
                for other_id in person_ids
                if other_id != person_id
            ]

            for original in original_list:
                sampled_people = self.random.sample(
                    other_person_ids,
                    min(5, len(other_person_ids))
                )

                for other_id in sampled_people:
                    other_original = self.random.choice(
                        originals[other_id]
                    )

                    negative_candidates.append(
                        (original, other_original, 0)
                    )

            # Pozitif ve negatif çiftleri dengeler.
            negative_count = min(
                len(negative_candidates),
                len(positive_pairs)
            )

            selected_negatives = self.random.sample(
                negative_candidates,
                negative_count
            )

            self.pairs.extend(positive_pairs)
            self.pairs.extend(selected_negatives)

        self.random.shuffle(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def apply_stroke_augmentation(self, image):
        image_np = np.array(image)

        if self.random.random() < 0.5:
            kernel = np.ones((2, 2), np.uint8)

            if self.random.random() < 0.5:
                image_np = cv2.erode(
                    image_np,
                    kernel,
                    iterations=1
                )
            else:
                image_np = cv2.dilate(
                    image_np,
                    kernel,
                    iterations=1
                )

        return Image.fromarray(image_np)

    def cached_preprocess(self, image_path):
        # Preprocess sonucunu diske cache'ler; varsa yeniden hesaplamaz.
        PREPROCESS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        key = hashlib.md5(str(image_path.resolve()).encode()).hexdigest()
        cache_file = PREPROCESS_CACHE_DIR / f"{key}.png"

        if not cache_file.exists():
            # Önce geçici dosyaya yaz, sonra atomik taşı: num_workers ile iki
            # süreç aynı anda yazsa bile bozuk/yarım dosya oluşmaz.
            with tempfile.NamedTemporaryFile(
                suffix=".png",
                dir=PREPROCESS_CACHE_DIR,
                delete=False
            ) as temp:
                temp_name = temp.name

            try:
                preprocess_signature(image_path, temp_name)

                os.replace(temp_name, cache_file)
            except Exception:
                if os.path.exists(temp_name):
                    os.remove(temp_name)
                raise

        return Image.open(cache_file).convert("L")

    def load_image(self, image_path):
        image = self.cached_preprocess(image_path)

        if self.augment:
            image = self.augment_transform(image)
            image = self.apply_stroke_augmentation(image)

        image = self.base_transform(image)

        return image

    def __getitem__(self, index):
        img1_path, img2_path, label = self.pairs[index]

        img1 = self.load_image(img1_path)
        img2 = self.load_image(img2_path)

        return (
            img1,
            img2,
            torch.tensor(label, dtype=torch.float32)
        )
