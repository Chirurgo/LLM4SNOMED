import json
import warnings
from typing import List, Tuple, Union
import numpy as np

class UndefinedMetricWarning(UserWarning):
    pass


def _warn(metric: str, result_size: int):
    msg = (
        f"{metric} is ill-defined and being set to 0.0 in samples with no predicted or true entities. "
        f"Use `zero_division` to control this behavior."
    )
    warnings.warn(msg, UndefinedMetricWarning, stacklevel=3)


def _flatten(
    y_true: List[List[Tuple[str, Tuple[int, int]]]],
    y_pred: List[List[Tuple[str, Tuple[int, int]]]]
):
    flat_true = []
    flat_pred = []
    for idx, (true_ents, pred_ents) in enumerate(zip(y_true, y_pred)):
        flat_true.extend([(label, span[0], span[1], idx) for label, span in true_ents])
        flat_pred.extend([(label, span[0], span[1], idx) for label, span in pred_ents])
    return flat_true, flat_pred


class DomainEvaluator:
    """
    Evaluate NER spans by label and start/end indices.
    Build from paths with NEREvaluator.from_paths(gold_path, pred_path).

    Each sample is a list of entities: [(label:str, (start:int, end:int)), ...]
    """

    def __init__(
        self,
        y_true: List[List[Tuple[str, Tuple[int, int]]]],
        y_pred: List[List[Tuple[str, Tuple[int, int]]]],
        *,
        to_lowercase_labels: bool = True,
        zero_division: Union[int, str] = "warn"
    ):
        self.zero_division = zero_division
        self.to_lowercase_labels = to_lowercase_labels
        self.y_true = self._normalize(y_true, source="gold")
        self.y_pred = self._normalize(y_pred, source="pred")
        self._align_lengths()

    # ---------- Constructors ----------
    @classmethod
    def from_paths(
        cls,
        gold_path: str,
        pred_path: str,
        *,
        to_lowercase_labels: bool = True,
        zero_division: Union[int, str] = "warn"
    ):
        y_true = cls._load_gold_entities(gold_path)
        y_pred = cls._load_pred_entities(pred_path)
        return cls(y_true, y_pred,
                   to_lowercase_labels=to_lowercase_labels,
                   zero_division=zero_division)

    @staticmethod
    def _load_gold_entities(path: str):
        with open(path, "r", encoding="utf-8") as f:
            annotated = json.load(f)

        all_gold = []
        for item in annotated:
            spans = []
            for ent in item.get("entity", []):
                start = int(ent["start_index"])
                end = int(ent["end_index"])
                # prefer "label", fallback to "domain" if someone got creative
                label_raw = ent.get("label", ent.get("domain", ""))
                label = "" if label_raw is None else str(label_raw)
                spans.append((label, (start, end)))
            all_gold.append(spans)
        return all_gold

    @staticmethod
    def _load_pred_entities(path: str):
        with open(path, "r", encoding="utf-8") as f:
            prediction_batches = json.load(f)

        all_preds = []
        for preds in prediction_batches:
            spans = []
            for ent in preds:
                start = int(ent["start_index"])
                end = int(ent["end_index"])
                # prefer "domain", fallback to "label" because why not be inconsistent twice
                label_raw = ent.get("domain", ent.get("label", ""))
                label = "" if label_raw is None else str(label_raw)
                spans.append((label, (start, end)))
            all_preds.append(spans)
        return all_preds

    # ---------- Normalization & checks ----------
    def _normalize(self, data, source: str):
        norm = []
        for sample in data:
            s = []
            for rec in sample:
                if not (isinstance(rec, (list, tuple)) and len(rec) == 2):
                    raise ValueError(f"{source}: expected (label, (start,end)), got {rec}")
                label, span = rec
                if not (isinstance(span, (list, tuple)) and len(span) == 2):
                    raise ValueError(f"{source}: bad span {span}")
                start, end = int(span[0]), int(span[1])
                lab = "" if label is None else str(label)
                if self.to_lowercase_labels:
                    lab = lab.lower()
                s.append((lab, (start, end)))
            norm.append(s)
        return norm

    def _align_lengths(self):
        n_true, n_pred = len(self.y_true), len(self.y_pred)
        if n_true != n_pred:
            warnings.warn(
                f"Gold/pred batches differ in length (gold={n_true}, pred={n_pred}). "
                f"Using min length = {min(n_true, n_pred)}.",
                UserWarning
            )
            m = min(n_true, n_pred)
            self.y_true = self.y_true[:m]
            self.y_pred = self.y_pred[:m]

    def _calculate_ci(self, tp, fn, fp, z=1.96):
        results = {}

        # 1. Standard Proportions (Precision, Recall)
        metrics_to_calc = [
            (tp, tp + fp, "Precision"),
            (tp, tp + fn, "Recall"),
        ]
        
        for x, n, name in metrics_to_calc:
            if n <= 0:
                results[name] = {"val": 0.0, "l95": 0.0, "u95": 0.0}
                continue
                
            p = x / n
            se = np.sqrt(p * (1.0 - p) / n)
            results[name] = {
                "val": round(float(p * 100), 2),
                "l95": round(float(max(0, p - z * se) * 100), 2),
                "u95": round(float(min(1, p + z * se) * 100), 2)
            }

        # 2. F1-Score Calculation
        # F1 = 2TP / (2TP + FP + FN)
        f1_num = 2 * tp
        f1_den = (2 * tp) + fp + fn
        
        if f1_den <= 0:
            results["F1"] = {"val": 0.0, "l95": 0.0, "u95": 0.0}
        else:
            f1_p = f1_num / f1_den
            # SE for F1 using the same proportion logic (Wald method)
            f1_se = np.sqrt(f1_p * (1.0 - f1_p) / f1_den)
            
            results["F1"] = {
                "val": round(float(f1_p * 100), 2),
                "l95": round(float(max(0, f1_p - z * f1_se) * 100), 2),
                "u95": round(float(min(1, f1_p + z * f1_se) * 100), 2)
            }
            
        return results
    # ---------- Metrics ----------
    def strict(self):
        gold, pred = _flatten(self.y_true, self.y_pred)
        gold_set = set(gold)
        pred_set = set(pred)

        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        ci = self._calculate_ci(tp, fn, fp)

        if (tp + fp) == 0 and (tp + fn) == 0:
            _warn("Precision/Recall/F1", result_size=0)
        elif (tp + fp) == 0:
            _warn("Precision", result_size=1)
        elif (tp + fn) == 0:
            _warn("Recall", result_size=1)

        print("Strict evaluation")
        for metric, values in ci.items():
                val = float(values['val'])
                l95 = float(values['l95'])  
                u95 = float(values['u95'])  
                print(f"{metric:10}: {val:>6.2f}% (95% CI: {l95:>6.2f}% – {u95:>6.2f}%)")

    @staticmethod
    def _is_overlap(span1, span2):
        return not (span1[1] <= span2[0] or span2[1] <= span1[0])

    def loose(self):
        gold, pred = _flatten(self.y_true, self.y_pred)
        gold_used = set()
        tp = 0

        for pred_ent in pred:
            pred_label, p_start, p_end, p_idx = pred_ent
            for gold_ent in gold:
                g_label, g_start, g_end, g_idx = gold_ent
                if (
                    g_idx == p_idx and
                    g_label == pred_label and
                    self._is_overlap((p_start, p_end), (g_start, g_end))
                ):
                    if gold_ent not in gold_used:
                        gold_used.add(gold_ent)
                        tp += 1
                        break

        fp = len(pred) - tp
        fn = len(gold) - tp

        ci = self._calculate_ci(tp, fn, fp)

        if (tp + fp) == 0 and (tp + fn) == 0:
            _warn("Precision/Recall/F1", result_size=0)
        elif (tp + fp) == 0:
            _warn("Precision", result_size=1)
        elif (tp + fn) == 0:
            _warn("Recall", result_size=1)

        print("Loose evaluation")
        for metric, values in ci.items():
                val = float(values['val'])
                l95 = float(values['l95'])
                u95 = float(values['u95'])
                print(f"{metric:10}: {val:>6.2f}% (95% CI: {l95:>6.2f}% – {u95:>6.2f}%)")

    # ---------- math helpers ----------
    def _safe_div(self, num: int, den: int) -> float:
        if den == 0:
            if self.zero_division in (0, "warn"):
                return 0.0
            return 1.0
        return num / den

    @staticmethod
    def _safe_f1(p: float, r: float) -> float:
        return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
