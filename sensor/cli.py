from __future__ import annotations
import argparse, json
from sensor.platform import train_csv

def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe sensor fault training")
    parser.add_argument("command", choices=["train"]); parser.add_argument("--source", choices=["csv"], default="csv")
    parser.add_argument("--path", required=True); parser.add_argument("--output-dir", default="artifacts/production")
    args = parser.parse_args()
    print(json.dumps(train_csv(args.path, args.output_dir), indent=2, default=str))
if __name__ == "__main__": main()
