import _bootstrap
import argparse
import json
from pathlib import Path
from polarity_d.evaluation import evaluate_fold, aggregate

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Recalculate Dice/Precision/Recall from native-volume predictions")
    p.add_argument("--model-folder", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--fold", type=int, required=True, choices=range(5))
    p.add_argument('--splits', help='Optional local five-fold splits JSON for older model folders')
    a = p.parse_args()
    folder = Path(a.model_folder) / f"fold_{a.fold}"
    if a.splits:
        cases = json.loads(Path(a.splits).read_text())[a.fold]['val']
    elif (folder / 'validation_cases.json').exists():
        cases = json.loads((folder / 'validation_cases.json').read_text())
    else:
        p.error('No validation case list found. Pass --splits /path/to/local/splits_final.json. Prediction-only weight packages do not contain validation data.')
    print(evaluate_fold(folder / "validation", a.labels, cases, a.fold, folder / "evaluation"))
    print(aggregate(a.model_folder))
