"""Create visual and automatic quality-control previews for preprocessed signatures.

The source dataset stays unchanged and every generated file goes into --out.

Output:
  qc_output/
    contact_sheets/<set>_NNN.png   # raw and processed images in a grid
    _SUSPECT/<TYPE>/<name>.png     # automatically flagged raw and processed pairs
    report.csv                     # metrics and flag for every image

Flags:
  EMPTY  ------> almost no ink remains after preprocessing
  SPARSE ------> very little ink remains so the signature may be faint or incomplete
  FRAME  ------> a long straight line remains and may belong to a frame or signature underline

Usage:
  python qc_preview.py
  python qc_preview.py --all
  python qc_preview.py --per-dataset 200 --out qc_output
"""

import argparse
import csv
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np

from preprocess import (
    preprocess_signature,
    TARGET_SIZE,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"} # File types accepted as signature images

EMPTY_INK_RATIO = 0.003 # Ink below 0.3% of all pixels represents an almost empty image
SPARSE_INK_RATIO = 0.010 # Ink below 1% represents a sparse or possibly incomplete signature
FRAME_LINE_RATIO = 0.65 # A straight line covering at least 65% of width or height may be a remaining frame


def list_images(root):
    return [
        p for p in sorted(root.rglob("*")) # p = Image path found by searching through every folder below root
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def dataset_name(path, root):
    rel = path.relative_to(root) # rel = Path without the dataset root ------> dataset/cedar/image.png becomes cedar/image.png

    return rel.parts[0] if len(rel.parts) > 1 else "root" # First folder represents the dataset name


def process_to_array(path, temp_path):
    preprocess_signature(path, temp_path) # Uses the same preprocessing route as training validation and the application

    return cv2.imread(str(temp_path), cv2.IMREAD_GRAYSCALE) # Returns the processed image as a one-channel NumPy array


def has_long_line(processed):
    ink = (processed < 128).astype(np.uint8) * 255 # Dark pixels become white in a binary mask ------> easier for OpenCV to detect
    h, w = ink.shape # h w = Image height and width

    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(int(w * FRAME_LINE_RATIO), 1), 1)) # hk = Kernel used to search for long horizontal lines
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(int(h * FRAME_LINE_RATIO), 1))) # vk = Kernel used to search for long vertical lines

    lines = cv2.bitwise_or(
        cv2.morphologyEx(ink, cv2.MORPH_OPEN, hk), # Morphological opening removes shapes that do not fit the line kernel
        cv2.morphologyEx(ink, cv2.MORPH_OPEN, vk)
    ) # bitwise_or combines detected horizontal and vertical lines

    return cv2.countNonZero(lines) > 0 # Any remaining white pixel means a long line was detected


def classify(processed):
    total = processed.shape[0] * processed.shape[1] # Total pixels = Height * Width
    ink_ratio = int((processed < 128).sum()) / total # Ink ratio = Dark pixels / All pixels

    if ink_ratio < EMPTY_INK_RATIO:
        return "EMPTY", ink_ratio

    if ink_ratio < SPARSE_INK_RATIO:
        return "SPARSE", ink_ratio

    if has_long_line(processed):
        return "FRAME", ink_ratio

    return "OK", ink_ratio


def side_by_side(raw_gray, processed):
    raw = cv2.resize(raw_gray, (TARGET_SIZE, TARGET_SIZE)) # Resizes the original image to match the processed 224x224 output
    sep = np.ones((TARGET_SIZE, 4), np.uint8) * 128 # sep = Four-pixel gray separator between both images

    return np.hstack([raw, sep, processed]) # hstack joins raw separator and processed arrays horizontally


def build_contact_sheet(pairs, cols, cell):
    thumb_w = cell * 2 + 4 # Thumbnail width = Two square images + Four-pixel separator
    tiles = [] # tiles = Labeled preview cells that will form the contact sheet

    for combined, label in pairs:
        t = cv2.resize(combined, (thumb_w, cell)) # t = Small raw and processed preview
        strip = np.ones((16, thumb_w), np.uint8) * 255 # White label area below the preview

        cv2.putText(strip, label[:42], (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 0, 1) # Limits long file names to 42 characters
        tiles.append(np.vstack([t, strip]))

    tile_h, tile_w = tiles[0].shape # Dimensions of one completed preview tile
    rows = [] # rows = Horizontal tile groups that will be stacked vertically

    for i in range(0, len(tiles), cols):
        chunk = tiles[i:i + cols] # chunk = Tiles belonging to the current contact-sheet row

        while len(chunk) < cols:
            chunk.append(np.ones((tile_h, tile_w), np.uint8) * 255) # Fills an incomplete final row with blank white tiles

        rows.append(np.hstack(chunk))

    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser(description = "Read-only preprocessing quality control.") # ArgumentParser reads options written after the script name
    ap.add_argument("--root", default = "dataset", help = "dataset root folder")
    ap.add_argument("--out", default = "qc_output", help = "output folder")
    ap.add_argument("--per-dataset", type = int, default = 400, help = "samples per dataset")
    ap.add_argument("--all", action = "store_true", help = "process every file without sampling")
    ap.add_argument("--cols", type = int, default = 6, help = "contact-sheet column count")
    ap.add_argument("--rows-per-sheet", type = int, default = 6, help = "rows per contact sheet")
    ap.add_argument("--cell", type = int, default = 112, help = "thumbnail height in pixels")
    ap.add_argument("--seed", type = int, default = 42, help = "random sampling seed")

    args = ap.parse_args() # args stores the values given by the user or their defaults

    root = Path(args.root) # Folder containing source datasets
    out = Path(args.out) # Folder that receives every generated file
    sheets_dir = out / "contact_sheets" # Contact sheets show several raw and processed pairs together
    suspect_dir = out / "_SUSPECT" # Automatically flagged previews are grouped here by flag

    sheets_dir.mkdir(parents = True, exist_ok = True) # Creates missing parent folders and does not fail when they already exist
    suspect_dir.mkdir(parents = True, exist_ok = True)

    rng = random.Random(args.seed) # Fixed seed makes random sampling repeatable

    all_images = list_images(root) # Every supported image found below the dataset root
    by_set = {} # Dictionary ------> dataset name maps to its image-path list

    for p in all_images:
        by_set.setdefault(dataset_name(p, root), []).append(p) # setdefault creates an empty list when the dataset name is new

    for name in by_set:
        files = by_set[name] # Images that belong to the current dataset

        if not args.all and len(files) > args.per_dataset:
            by_set[name] = sorted(rng.sample(files, args.per_dataset)) # Selects a repeatable random sample without duplicates

    per_sheet = args.cols * args.rows_per_sheet # Images per sheet = Columns * Rows
    report_rows = [] # One CSV row for every processed or failed image
    counts = {"OK": 0, "EMPTY": 0, "SPARSE": 0, "FRAME": 0, "ERROR": 0} # Running total for every quality flag

    with tempfile.TemporaryDirectory() as td:
        temp_path = Path(td) / "proc.png" # Temporary processed image is deleted automatically when this block ends

        for name, files in by_set.items():
            sheet_pairs = [] # Raw and processed previews waiting to be written into a contact sheet
            sheet_index = 1 # Number added to each contact-sheet file name

            print(f"[{name}] Processing {len(files)} files...")

            for path in files:
                raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) # Loads the source image as one-channel grayscale

                if raw is None:
                    counts["ERROR"] += 1
                    report_rows.append([str(path), name, "", "READ_FAIL"])
                    continue

                try:
                    processed = process_to_array(path, temp_path) # Processed image produced by the real application pipeline
                    label, ink_ratio = classify(processed) # label = Quality flag | ink_ratio = Dark-pixel percentage as a value from 0 to 1
                except Exception as exc: # noqa: BLE001 ------> one broken image should not stop the complete quality check
                    counts["ERROR"] += 1
                    report_rows.append([str(path), name, "", f"ERROR:{exc}"])
                    continue

                counts[label] += 1 # Increases the total of the detected quality class
                report_rows.append([str(path), name, f"{ink_ratio:.4f}", label])

                combined = side_by_side(raw, processed) # Raw image on the left and processed image on the right

                if label != "OK":
                    tdir = suspect_dir / label # tdir = Folder for the detected suspect type
                    tdir.mkdir(exist_ok = True)

                    flat = str(path.relative_to(root)).replace("/", "__") # Flattens nested folders into one safe output file name
                    cv2.imwrite(str(tdir / f"{flat}.png"), combined) # Saves the raw and processed comparison for visual review

                sheet_pairs.append((combined, path.stem))

                if len(sheet_pairs) == per_sheet:
                    sheet = build_contact_sheet(sheet_pairs, args.cols, args.cell)
                    cv2.imwrite(str(sheets_dir / f"{name}_{sheet_index:03d}.png"), sheet)

                    sheet_pairs = []
                    sheet_index += 1

            if sheet_pairs: # Saves remaining previews when they do not fill a complete final sheet
                sheet = build_contact_sheet(sheet_pairs, args.cols, args.cell)
                cv2.imwrite(str(sheets_dir / f"{name}_{sheet_index:03d}.png"), sheet)

    with open(out / "report.csv", "w", newline = "") as f: # newline prevents blank lines between CSV rows on some systems
        w = csv.writer(f) # CSV writer converts each report list into one comma-separated row

        w.writerow(["path", "dataset", "ink_ratio", "flag"])
        w.writerows(report_rows)

    total = sum(counts.values()) # Total number of checked images including failures

    print("\n=== SUMMARY ===")

    for k, v in counts.items():
        pct = (100 * v / total) if total else 0 # Percentage = Class count / All checked images * 100
        print(f"  {k:6s}: {v:5d}  ({pct:.1f}%)")

    print(f"\nContact sheets : {sheets_dir}")
    print(f"Suspects       : {suspect_dir}  (EMPTY/SPARSE/FRAME)")
    print(f"Report         : {out / 'report.csv'}")
    print("\nNote: FRAME suspects may include the signature's own underline ------> confirm them visually.")


if __name__ == "__main__": # Runs main only when this file is started directly
    main()
