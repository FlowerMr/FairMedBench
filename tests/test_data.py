from pathlib import Path
import pandas as pd
from PIL import Image
from fairmedbench.data.ddi import build_ddi_records


def test_ddi_record_builder(tmp_path: Path):
    image = tmp_path / "x.png"
    Image.new("RGB", (8, 8)).save(image)
    metadata = tmp_path / "meta.csv"
    pd.DataFrame([{"DDI_file": "x.png", "malig": 1, "skin_tone": "I-II"}]).to_csv(metadata, index=False)
    records = build_ddi_records(metadata, tmp_path)
    assert len(records) == 1
    assert records[0].label == 1
    assert records[0].subgroup == "I-II"
