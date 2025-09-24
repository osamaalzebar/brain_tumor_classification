import csv
import shutil
from pathlib import Path

# Root directory containing folders 1,2,3 and the 'train' folder
ROOT = Path("/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/densnet201/data")

SOURCE_LABELS = ["1", "2", "3"]            # folder names used as labels
TRAIN_DIR = ROOT / "train"                 # destination directory
CSV_PATH = ROOT / "image_labels.csv"       # CSV output path

# Define which file extensions to treat as images
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp"}

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def unique_name(dest_dir: Path, filename: str) -> str:
    """
    If filename already exists in dest_dir, append _1, _2, ... before the extension.
    Returns a unique filename string.
    """
    candidate = dest_dir / filename
    if not candidate.exists():
        return filename

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    i = 1
    while True:
        new_name = f"{stem}_{i}{suffix}"
        if not (dest_dir / new_name).exists():
            return new_name
        i += 1

def main():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    # Open CSV and write header
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["image_path", "label"])  # header

        moved_count = 0
        for label in SOURCE_LABELS:
            src_dir = ROOT / label
            if not src_dir.exists():
                print(f"[WARN] Source folder not found: {src_dir} (skipping)")
                continue

            for item in sorted(src_dir.iterdir()):
                if not is_image(item):
                    continue

                # Resolve name conflicts in train/
                final_name = unique_name(TRAIN_DIR, item.name)
                dest_path = TRAIN_DIR / final_name

                # Move the file
                shutil.move(str(item), str(dest_path))

                # Write CSV row: final filename (only the name), original label
                writer.writerow([final_name, label])
                moved_count += 1

        print(f"[DONE] Moved {moved_count} images into '{TRAIN_DIR.name}' "
              f"and wrote labels to '{CSV_PATH.name}'.")

if __name__ == "__main__":
    main()
