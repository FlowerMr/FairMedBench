import json
from pathlib import Path
import pandas as pd


def save_report(
    output_dir,
    overall,
    subgroup,
    fairness,
    calibration,
    robustness=None,
    xai=None,
    uncertainty=None,
    notes=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([overall]).to_csv(output_dir / "overall_metrics.csv", index=False)
    subgroup.to_csv(output_dir / "subgroup_metrics.csv", index=False)
    pd.DataFrame([fairness]).to_csv(output_dir / "fairness_metrics.csv", index=False)
    pd.DataFrame([calibration]).to_csv(output_dir / "calibration_metrics.csv", index=False)
    if robustness is not None:
        pd.DataFrame([robustness]).to_csv(output_dir / "robustness_metrics.csv", index=False)
    if xai is not None:
        pd.DataFrame(xai).to_csv(output_dir / "xai_metrics.csv", index=False)
    if uncertainty is not None:
        pd.DataFrame(uncertainty).to_csv(output_dir / "uncertainty_metrics.csv", index=False)
    payload = {
        "overall": overall,
        "subgroup": subgroup.to_dict(orient="records"),
        "fairness": fairness,
        "calibration": calibration,
        "robustness": robustness,
        "xai": xai,
        "uncertainty": uncertainty,
        "notes": notes or [],
    }
    (output_dir / "report.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
