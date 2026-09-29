from pathlib import Path
import pandas as pd

from .base import ImageRecord


def _find_column(frame: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    normalized = {str(c).strip().lower(): c for c in frame.columns}

    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    raise ValueError(
        f"Could not identify the {purpose} column. "
        f"Available columns: {list(frame.columns)}"
    )


def load_ddi_metadata(metadata_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(metadata_path)

    image_col = _find_column(
        frame,
        ["DDI_file", "image_id", "image", "filename", "file"],
        "image"
    )

    # DDI metadata uses "malignant" as the binary label column.
    label_col = _find_column(
        frame,
        ["malig", "malignancy", "malignant", "label", "target"],
        "binary label"
    )

    subgroup_col = _find_column(
        frame,
        ["skin_tone", "skin_type", "fst", "fitzpatrick"],
        "skin-tone subgroup"
    )

    result = frame.rename(
        columns={
            image_col: "image_id",
            label_col: "label",
            subgroup_col: "subgroup",
        }
    ).copy()

    result["label"] = result["label"].astype(int)
    result["subgroup"] = result["subgroup"].astype(str)
    result["image_id"] = result["image_id"].astype(str)

    return result


def build_ddi_records(
    metadata_path: str | Path,
    image_root: str | Path
) -> list[ImageRecord]:

    metadata = load_ddi_metadata(metadata_path)
    image_root = Path(image_root)

    records = []
    extensions = [".png", ".jpg", ".jpeg", ".webp"]

    for row in metadata.itertuples(index=False):
        stem = Path(row.image_id).stem

        candidates = [image_root / row.image_id]
        candidates.extend(
            image_root / f"{stem}{ext}"
            for ext in extensions
        )

        image_path = next(
            (p for p in candidates if p.exists()),
            None
        )

        if image_path is None:
            raise FileNotFoundError(
                f"Image not found for metadata id "
                f"'{row.image_id}' under {image_root}"
            )

        records.append(
            ImageRecord(
                str(image_path),
                int(row.label),
                str(row.subgroup),
                stem
            )
        )

    return records