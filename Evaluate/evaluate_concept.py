import json
import warnings
import numpy as np
from collections import defaultdict
from typing import List, Literal, Tuple, Union, Dict, Set, Iterable, Optional
from neo4j import GraphDatabase


class UndefinedMetricWarning(UserWarning):
    pass


def _warn_prf(average: str, modifier: str, msg_start: str, result_size: int):
    axis0, axis1 = ("label", "sample") if average == "samples" else ("sample", "label")
    msg = (
        f"{msg_start} ill-defined and being set to 0.0 in {axis1}s with no {modifier} {axis0}s."
        if result_size > 1
        else f"{msg_start} ill-defined and being set to 0.0 due to no {modifier} {axis0}."
    )
    msg += " Use `zero_division` parameter to control this behavior."
    warnings.warn(msg, UndefinedMetricWarning, stacklevel=3)


def _prf_divide(numerator: np.ndarray,
                denominator: np.ndarray,
                metric: Literal["precision", "recall", "f-score"],
                modifier: str,
                average: str,
                warn_for: List[str],
                zero_division: Union[str, int] = "warn") -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.true_divide(numerator, denominator)
        result[denominator == 0] = 0.0 if zero_division in ["warn", 0] else 1.0

    if (denominator == 0).any() and zero_division == "warn" and metric in warn_for:
        msg_start = f"{metric.title()}"
        if "f-score" in warn_for:
            msg_start += " and F-score" if metric in warn_for else "F-score"
        msg_start += " are" if "f-score" in warn_for else " is"
        _warn_prf(average=average, modifier=modifier, msg_start=msg_start, result_size=len(result))
    return result


def flatten_for_eval(y_true, y_pred):
    all_true = []
    all_pred = []
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        all_true.extend([t + [i] for t in true])
        all_pred.extend([p + [i] for p in pred])
    return all_true, all_pred


def extract_tp_actual_correct(y_true, y_pred):
    entities_true = defaultdict(set)
    entities_pred = defaultdict(set)

    for type_name, (start, end), concept_id, idx in y_true:
        entities_true[type_name].add((start, end, concept_id, idx))
    for type_name, (start, end), concept_id, idx in y_pred:
        entities_pred[type_name].add((start, end, concept_id, idx))

    target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))

    tp_sum = np.array([], dtype=np.int32)
    pred_sum = np.array([], dtype=np.int32)
    true_sum = np.array([], dtype=np.int32)
    for type_name in target_names:
        gold = entities_true[type_name]
        pred = entities_pred[type_name]
        tp = len(gold & pred)
        tp_sum = np.append(tp_sum, tp)
        pred_sum = np.append(pred_sum, len(pred))
        true_sum = np.append(true_sum, len(gold))

    return pred_sum, tp_sum, true_sum, target_names


def _calculate_ci(tp, fn, fp, z=1.96):
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


def compute_prf(y_true, y_pred, average="micro", zero_division: Union[str, int] = "warn"):
    y_true, y_pred = flatten_for_eval(y_true, y_pred)
    pred_sum, tp_sum, true_sum, _ = extract_tp_actual_correct(y_true, y_pred)

    if average == "micro":
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

    precision = _prf_divide(tp_sum, pred_sum, "precision", "predicted", average,
                            ["precision", "recall", "f-score"], zero_division=zero_division)
    recall = _prf_divide(tp_sum, true_sum, "recall", "true", average,
                         ["precision", "recall", "f-score"], zero_division=zero_division)
    f1 = 2 * (precision * recall) / np.maximum(precision + recall, 1e-8)
    ci = _calculate_ci(tp_sum[0], true_sum[0] - tp_sum[0], pred_sum[0] - tp_sum[0])

    return {"precision": precision[0], "recall": recall[0], "f_score": f1[0], "ci": ci}


def extract_gold_concept_ids(
    all_true: List[List[List[object]]]
) -> Set[str]:
    """
    all_true:
        List[ List[ ["concept", (start, end), concept_id:str] ] ]

    Return set of all concept_ids appearing in gold annotations.
    """
    concept_ids: Set[str] = set()
    for doc in all_true:
        for _, _, cid in doc:
            concept_ids.add(str(cid))
    return concept_ids


def build_concept_proximity_from_neo4j(
    uri: str,
    user: str,
    password: str,
    gold_concept_ids: Iterable[str],
) -> Dict[str, Set[str]]:
    """
    For each gold concept_id, fetch related concepts in Neo4j via
    HAS_ACTIVE_INGREDIENT and IS_A, and build:

        gold_id(str) -> {gold_id, related_1, related_2, ...}  (all as str)
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))

    query = """
    MATCH (p:Concept)-[r:HAS_ACTIVE_INGREDIENT|IS_A]-(c:Concept)
    WHERE toString(p.concept_id) IN $concept_ids
    RETURN toString(p.concept_id) AS gold_id,
           collect(DISTINCT toString(c.concept_id)) AS neighbors
    """

    gold_concept_ids = [str(cid) for cid in gold_concept_ids]
    proximity: Dict[str, Set[str]] = {cid: {cid} for cid in gold_concept_ids}

    try:
        with driver.session() as session:
            result = session.run(query, concept_ids=gold_concept_ids)
            for record in result:
                gold_id = record["gold_id"]
                neighbors = set(record["neighbors"])
                neighbors.add(gold_id)  # include self as acceptable
                proximity[gold_id] = neighbors
    finally:
        driver.close()

    return proximity



class Evaluator:
    """
    Evaluator that can be built either from in-memory data (lists of spans)
    or directly from file paths via Evaluator.from_paths(gold_path, pred_path).

    Each item is a list of entities: ["concept", (start, end), concept_id]
    """
    def __init__(self,
                 all_true: List[List[List[Union[str, Tuple[int, int], str]]]],
                 all_outs: List[List[List[Union[str, Tuple[int, int], str]]]],
                 concept_proximity: Optional[Dict[str, Set[str]]] = None,
                 ):
        self.all_true = self._normalize(all_true, source="gold")
        self.all_outs = self._normalize(all_outs, source="pred")
        self._sanity_align_lengths()
        self.concept_proximity = concept_proximity or {}

    # ---------- construction helpers ----------
    @classmethod
    def from_paths(cls, gold_path: str, pred_path: str):
        all_true = cls._load_gold_concept_data(gold_path)
        all_outs = cls._load_predicted_concepts(pred_path)
        return cls(all_true, all_outs)

    @classmethod
    def from_neo4j(
        cls,
        gold_path: str,
        pred_path: str,
        uri: str,
        user: str,
        password: str,
    ):
        """
        Build Evaluator from JSON paths and populate concept_proximity
        using Neo4j (HAS_ACTIVE_INGREDIENT, IS_A).
        """
        all_true = cls._load_gold_concept_data(gold_path)
        all_outs = cls._load_predicted_concepts(pred_path)

        gold_ids = extract_gold_concept_ids(all_true)
        concept_proximity = build_concept_proximity_from_neo4j(
            uri=uri,
            user=user,
            password=password,
            gold_concept_ids=gold_ids,
        )

        return cls(all_true, all_outs, concept_proximity=concept_proximity)

    @staticmethod
    def _load_gold_concept_data(path: str):
        with open(path, "r", encoding="utf-8") as f:
            annotated = json.load(f)

        all_gold = []
        for item in annotated:
            spans = []
            for ent in item.get("entity", []):
                spans.append([
                    "concept",
                    (int(ent["start_index"]), int(ent["end_index"])),
                    str(ent["Target code"])
                ])
            all_gold.append(spans)
        return all_gold

    @staticmethod
    def _load_predicted_concepts(path: str):
        with open(path, "r", encoding="utf-8") as f:
            prediction_batches = json.load(f)

        all_preds = []
        for preds in prediction_batches:
            spans = []
            for ent in preds:
                spans.append([
                    "concept",
                    (int(ent["start_index"]), int(ent["end_index"])),
                    str(ent["concept_id"])
                ])
            all_preds.append(spans)
        return all_preds

    # ---------- normalization & checks ----------
    @staticmethod
    def _normalize(data, source: str):
        # Enforce: ["concept", (start, end), concept_id:str]
        norm = []
        for sample in data:
            s_spans = []
            for rec in sample:
                if len(rec) != 3:
                    raise ValueError(f"{source}: expected [type, (start,end), concept_id], got {rec}")
                typ, span, cid = rec
                if not isinstance(span, (list, tuple)) or len(span) != 2:
                    raise ValueError(f"{source}: bad span {span}")
                start, end = int(span[0]), int(span[1])
                s_spans.append([str(typ), (start, end), str(cid)])
            norm.append(s_spans)
        return norm

    def _sanity_align_lengths(self):
        n_true, n_pred = len(self.all_true), len(self.all_outs)
        if n_true != n_pred:
            warnings.warn(
                f"Gold/pred batches differ in length (gold={n_true}, pred={n_pred}). "
                f"Using min length = {min(n_true, n_pred)}.",
                UserWarning
            )
            m = min(n_true, n_pred)
            self.all_true = self.all_true[:m]
            self.all_outs = self.all_outs[:m]

    def _is_concept_close(self, gold_cid: str, pred_cid: str) -> bool:
        """
        True if pred_cid is acceptable for gold_cid:
        - exact match; or
        - in the proximity set for that gold concept.
        """
        if gold_cid == pred_cid:
            return True
        neighbors = self.concept_proximity.get(gold_cid)
        return neighbors is not None and pred_cid in neighbors

    def transform_data(self):
        return self.all_true, self.all_outs

    def strict(self, average: str = "micro", zero_division: Union[str, int] = "warn"):
        all_true_typed, all_outs_typed = self.transform_data()
        scores = compute_prf(all_true_typed, all_outs_typed, average=average, zero_division=zero_division)
        ci = scores['ci']
        print("Strict evaluation")
        for metric, values in ci.items():
            val = float(values['val'])
            l95 = float(values['l95'])  
            u95 = float(values['u95'])  
            print(f"{metric:10}: {val:>6.2f}% (95% CI: {l95:>6.2f}% – {u95:>6.2f}%)")

    def _is_overlap(self, span1, span2):
        return not (span1[1] <= span2[0] or span2[1] <= span1[0])

    def loose(self):
        all_true, all_pred = self.transform_data()

        gold = []
        pred = []

        for idx, (true_ents, pred_ents) in enumerate(zip(all_true, all_pred)):
            gold.extend([(start, end, concept_id, idx) for _, (start, end), concept_id in true_ents])
            pred.extend([(start, end, concept_id, idx) for _, (start, end), concept_id in pred_ents])

        tp = 0
        matched_gold = set()

        for p_start, p_end, p_cid, p_idx in pred:
            for g_start, g_end, g_cid, g_idx in gold:
                if (
                    g_idx == p_idx and
                    g_cid == p_cid and
                    not (p_end <= g_start or g_end <= p_start) and
                    (g_start, g_end, g_cid, g_idx) not in matched_gold
                ):
                    matched_gold.add((g_start, g_end, g_cid, g_idx))
                    tp += 1
                    break

        fp = len(pred) - tp
        fn = len(gold) - tp

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        ci = _calculate_ci(tp, fn, fp)
        print("Loose evaluation")
        for metric, values in ci.items():
            val = float(values['val'])
            l95 = float(values['l95'])  
            u95 = float(values['u95'])
            print(f"{metric:10}: {val:>6.2f}% (95% CI: {l95:>6.2f}% – {u95:>6.2f}%)")
    
    def hierarchy_aware(self):
        """
        Hierarchy-aware mode:

        - Span: requires overlap (like loose()).
        - Concept: TP if predicted concept is:
            a) exactly the gold concept_id, OR
            b) in the proximity set for that gold concept_id
               (IS_A / HAS_ACTIVE_INGREDIENT neighbors, etc.).
        """
        all_true, all_pred = self.transform_data()

        gold = []
        pred = []

        for idx, (true_ents, pred_ents) in enumerate(zip(all_true, all_pred)):
            gold.extend([(start, end, concept_id, idx) for _, (start, end), concept_id in true_ents])
            pred.extend([(start, end, concept_id, idx) for _, (start, end), concept_id in pred_ents])

        tp = 0
        matched_gold = set()

        for p_start, p_end, p_cid, p_idx in pred:
            p_span = (p_start, p_end)
            p_cid = str(p_cid)
            for g_start, g_end, g_cid, g_idx in gold:
                g_span = (g_start, g_end)
                g_cid = str(g_cid)
                if (
                    g_idx == p_idx and
                    self._is_concept_close(g_cid, p_cid) and
                    self._is_overlap(p_span, g_span) and
                    (g_start, g_end, g_cid, g_idx) not in matched_gold
                ):
                    matched_gold.add((g_start, g_end, g_cid, g_idx))
                    tp += 1
                    break

        fp = len(pred) - tp
        fn = len(gold) - tp

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        print("Hierarchy-aware evaluation")
        ci = _calculate_ci(tp, fn, fp)
        for metric, values in ci.items():
            val = float(values['val'])
            l95 = float(values['l95'])  
            u95 = float(values['u95'])
            print(f"{metric:10}: {val:>6.2f}% (95% CI: {l95:>6.2f}% – {u95:>6.2f}%)")

    def _span_iou(self, span1, span2):
        inter_start = max(span1[0], span2[0])
        inter_end = min(span1[1], span2[1])
        intersection = max(0, inter_end - inter_start)
        union = max(span1[1], span2[1]) - min(span1[0], span2[0])
        return intersection / union if union > 0 else 0.0

    def iou(self):
        all_true, all_pred = self.transform_data()
        
        # We need to track character-level sets for each concept per document
        # Using a dictionary: (doc_idx, concept_id) -> set of character indices
        gold_chars = defaultdict(set)
        pred_chars = defaultdict(set)
        concept_ids = set()

        for idx, (true_ents, pred_ents) in enumerate(zip(all_true, all_pred)):
            for _, (start, end), cid in true_ents:
                gold_chars[(idx, cid)].update(range(start, end))
                concept_ids.add(cid)
            for _, (start, end), cid in pred_ents:
                pred_chars[(idx, cid)].update(range(start, end))
                concept_ids.add(cid)

        total_iou = 0.0
        num_docs = len(all_true)
        
        # The paper calculates IoU per concept per document
        # and then takes the mean across documents
        for doc_idx in range(num_docs):
            doc_concepts = {cid for (i, cid) in (set(gold_chars.keys()) | set(pred_chars.keys())) if i == doc_idx}
            
            doc_iou_sum = 0.0
            if not doc_concepts:
                total_iou += 1.0 # Or 0.0 depending on how you handle empty docs; usually 1.0 if perfectly empty
                continue
                
            for cid in doc_concepts:
                g_set = gold_chars[(doc_idx, cid)]
                p_set = pred_chars[(doc_idx, cid)]
                
                intersection = len(g_set & p_set)
                union = len(g_set | p_set)
                
                if union > 0:
                    doc_iou_sum += (intersection / union)
            
            # Mean IoU for this document across all concepts present in ground truth or pred
            total_iou += (doc_iou_sum / len(doc_concepts))

        final_mean_iou = total_iou / num_docs
        report = f"Mean IoU: {final_mean_iou:.2%}"
        print("IoU evaluation")
        print(f"{'IoU':10}: {final_mean_iou*100:>6.2f}%")
        return report, final_mean_iou