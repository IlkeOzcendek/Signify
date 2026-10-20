import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from preprocess import preprocess_signature
from utils.enrollment import decide_registered_person
from utils.image_quality import inspect_uploaded_signature
from utils.model_assets import missing_model_assets


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class UploadedImage(io.BytesIO):
    def __init__(self, data, name="synthetic.png"):
        super().__init__(data)
        self.name = name


def image_upload(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return UploadedImage(stream.getvalue())


class PreprocessingTests(unittest.TestCase):
    def test_blank_image_is_rejected(self):
        quality = inspect_uploaded_signature(image_upload(Image.new("RGB", (640, 320), "white")))
        self.assertFalse(quality.usable)
        self.assertEqual(quality.processed_image.size, (224, 224))
        self.assertGreater(len(quality.errors), 0)

    def test_invalid_image_is_rejected_without_decoder_exception(self):
        quality = inspect_uploaded_signature(UploadedImage(b"not an image"))
        self.assertFalse(quality.usable)
        self.assertGreater(len(quality.errors), 0)

    def test_synthetic_stroke_is_centered_without_changing_source(self):
        canvas = np.full((320, 640, 3), 255, dtype=np.uint8)
        x = np.linspace(100, 540, 300).astype(np.int32)
        y = (160 + 35 * np.sin(x / 25)).astype(np.int32)
        cv2.polylines(canvas, [np.column_stack((x, y))], False, (20, 20, 20), 4)
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "synthetic.png"
            output = Path(temporary) / "processed.png"
            cv2.imwrite(str(source), canvas)
            original = source.read_bytes()
            for mode in ("default", "sharp"):
                preprocess_signature(source, output, mode=mode)
                result = cv2.imread(str(output), cv2.IMREAD_GRAYSCALE)
                self.assertEqual(result.shape, (224, 224))
                ink_y, ink_x = np.where(result < 128)
                self.assertGreater(len(ink_x), 50)
                self.assertGreater(ink_x.min(), 5)
                self.assertLess(ink_x.max(), 218)
                self.assertLess(abs((ink_x.min() + ink_x.max()) / 2 - 111.5), 5)
                self.assertGreater(ink_y.min(), 5)
            self.assertEqual(source.read_bytes(), original)


class ConsensusTests(unittest.TestCase):
    def decision(self, distances):
        return decide_registered_person(distances, threshold=0.6, confident_diff=0.9, confident_same=0.4)

    def test_one_lucky_reference_cannot_override_four_mismatches(self):
        self.assertEqual(self.decision([0.05, 1.1, 1.2, 1.3, 1.4])["verdict"], "no_match")

    def test_ambiguous_group_abstains(self):
        self.assertEqual(self.decision([0.2, 0.3, 0.5, 0.8, 1.2])["verdict"], "uncertain")

    def test_consistent_group_can_match_despite_one_outlier(self):
        self.assertEqual(self.decision([0.15, 0.2, 0.25, 0.3, 1.4])["verdict"], "match")

    def test_reference_order_does_not_change_decision(self):
        values = [0.15, 0.2, 0.25, 0.3, 1.4]
        self.assertEqual(self.decision(values), self.decision(list(reversed(values))))

    def test_no_reference_has_no_decision(self):
        with self.assertRaises(ValueError):
            self.decision([])


class ModelAvailabilityTests(unittest.TestCase):
    def test_general_requires_weights_and_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("verify.MODEL_PATH", "models/example.pth"):
                self.assertEqual(len(missing_model_assets("general", root)), 2)
                (root / "models").mkdir()
                (root / "models/example.pth").write_bytes(b"availability fixture")
                self.assertEqual(missing_model_assets("general", root), ["best_threshold.json"])
                (root / "best_threshold.json").write_text("{}")
                self.assertEqual(missing_model_assets("general", root), [])

    def test_archive_needs_both_heads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "models").mkdir()
            (root / "models/archive_v2_binary_head.pth").write_bytes(b"availability fixture")
            self.assertEqual(missing_model_assets("archive", root), ["models/archive_v2_gray_head.pth"])

    def test_unknown_engine_is_rejected(self):
        with self.assertRaises(ValueError):
            missing_model_assets("unknown")


class AppSmokeTests(unittest.TestCase):
    def test_fresh_install_pages_and_missing_model_guidance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ("app_pages", "utils", "db", "src", "scripts", "assets"):
                shutil.copytree(
                    PROJECT_ROOT / directory,
                    root / directory,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
            for filename in ("app.py", "preprocess.py", "verify.py", "verify_archive_v2.py", "verify_utsig_v3.py"):
                shutil.copy2(PROJECT_ROOT / filename, root / filename)
            code = """
import json
from streamlit.testing.v1 import AppTest
app = AppTest.from_file("app.py", default_timeout=60).run()
assert not app.exception, str(app.exception)
for page in ("admin", "analysis", "about", "verify"):
    app.sidebar.radio(key="navigation").set_value(page).run()
    assert not app.exception, str(app.exception)
for engine in ("general", "archive", "utsig"):
    check = AppTest.from_string(
        "from utils.model_assets import ensure_engine_ready\\n"
        "ensure_engine_ready(" + repr(engine) + ")"
    ).run()
    assert not check.exception, str(check.exception)
    assert check.warning, "Missing model guidance was not shown"
    assert "not installed" in check.warning[0].value
    assert check.code, "Missing asset names were not shown"
print(json.dumps({"pages": 4, "missing_model_guards": 3}))
"""
            environment = os.environ.copy()
            environment.update({
                "PYTHONPATH": str(root),
                "PYTHONDONTWRITEBYTECODE": "1",
                "TMPDIR": str(root),
                "MPLCONFIGDIR": str(root / "matplotlib"),
            })
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=root,
                env=environment,
                text=True,
                capture_output=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('"missing_model_guards": 3', result.stdout)


if __name__ == "__main__":
    unittest.main()
