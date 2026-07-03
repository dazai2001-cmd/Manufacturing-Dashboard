from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from manufacturing_dashboard.training import train_and_save_artifacts


def main():
    report = train_and_save_artifacts()
    metrics = report["test_metrics"]
    print(f"Saved {report['model_name']} model artifact.")
    print(f"Threshold: {report['chosen_threshold']}")
    print(
        "Held-out test metrics: "
        f"accuracy={metrics['accuracy']:.3f}, "
        f"precision={metrics['precision']:.3f}, "
        f"recall={metrics['recall']:.3f}, "
        f"f2={metrics['f2']:.3f}, "
        f"roc_auc={metrics['roc_auc']:.3f}"
    )


if __name__ == "__main__":
    main()
