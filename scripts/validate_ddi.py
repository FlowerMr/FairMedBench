import argparse
from fairmedbench.data.ddi import build_ddi_records

parser = argparse.ArgumentParser()
parser.add_argument("--metadata", required=True)
parser.add_argument("--images", required=True)
args = parser.parse_args()
records = build_ddi_records(args.metadata, args.images)
print(f"Validated {len(records)} DDI records.")
