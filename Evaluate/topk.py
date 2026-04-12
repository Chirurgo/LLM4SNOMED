
def concept_accuracy(gold_data, pred_data):
    """
    Evaluate strict accuracy of concept_id predictions from LLM.
    Only compares entities with exact span match.

    Args:
        gold_data: list of dicts with 'entity' and 'Target code'
        pred_data: list of list of predicted entities with 'concept_id'

    Returns:
        float: accuracy (correct / total matched)
    """
    assert len(gold_data) == len(pred_data), "Mismatch in number of records."

    total = 0
    correct = 0

    for gold_item, pred_entities in zip(gold_data, pred_data):
        gold_ents = {
            (e["start_index"], e["end_index"]): str(e["Target code"])
            for e in gold_item.get("entity", [])
        }

        for pred in pred_entities:
            span = (pred["start_index"], pred["end_index"])
            if span not in gold_ents:
                continue  # only match exact span

            gold_cid = gold_ents[span]
            pred_cid = str(pred.get("concept_id", ""))

            total += 1
            if pred_cid == gold_cid:
                correct += 1

    accuracy = correct / total if total else 0.0
    return accuracy


def evaluate_concept_precision_recall_f1(gold_data, pred_data):
    """
    Computes precision, recall, and F1 for concept prediction.
    Match = (start_index, end_index, concept_id)

    Returns:
        dict: {'precision': float, 'recall': float, 'f1': float}
    """
    assert len(gold_data) == len(pred_data), "Mismatch in number of records."

    gold_set = set()
    pred_set = set()

    for idx, (gold_item, pred_entities) in enumerate(zip(gold_data, pred_data)):
        for g in gold_item.get("entity", []):
            span = (g["start_index"], g["end_index"])
            gold_cid = str(g["Target code"])
            gold_set.add((span[0], span[1], gold_cid, idx))

        for p in pred_entities:
            span = (p["start_index"], p["end_index"])
            pred_cid = str(p.get("concept_id", ""))
            pred_set.add((span[0], span[1], pred_cid, idx))

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def topk_accuracy(gold_data, pred_data, top_k_values=[1, 5, 10]):
    """
    Evaluate top-k concept ranking accuracy from predicted candidates.
    Match is based on span (start, end) only.

    Args:
        gold_data: list of dicts with 'entity' and 'Target code'
        pred_data: list of list of predicted entities with 'tops'
        top_k_values: which k values to report (default: [1, 5, 10])

    Returns:
        dict with top-k accuracy scores
    """
    assert len(gold_data) == len(pred_data), "Mismatch in number of records."

    topk_hits = {k: 0 for k in top_k_values}
    total = 0
    empty_list_count = 0 

    for gold_item, pred_entities in zip(gold_data, pred_data):
        gold_ents = {
            (e["start_index"], e["end_index"]): str(e["Target code"])
            for e in gold_item.get("entity", [])
        }
        pred_entities = pred_entities or []
        for pred in pred_entities:
            span = (pred["start_index"], pred["end_index"])
            if span not in gold_ents:
                continue  # Only evaluate matching spans

            gold_cid = gold_ents[span]
            tops = pred.get("tops")

            # Check if tops is None or an empty list
            if not tops:
                # If tops is None or empty, set the candidates to ["Unknown"]
                empty_list_count += 1
                top_candidates = ["Unknown"]
            else:
                # If tops has candidates, process them.
                # Use a list comprehension that defensively handles 'None' for concept_id,
                # replacing it with "Unknown" if found.
                top_candidates = [
                    str(t.get("concept_id")) if t and t.get("concept_id") is not None else "Unknown" 
                    for t in tops
                ]

            total += 1
            for k in top_k_values:
                if gold_cid in top_candidates[:k]:
                    topk_hits[k] += 1
    empty_list_percentage = (empty_list_count / total) * 100 if total else 0.0

    results = {
        f"top{k}": topk_hits[k] / total if total else 0.0
        for k in top_k_values
    }
    print(f"No. of concept: {total}")
    print(f"No. of no mappable concept: {empty_list_count}({empty_list_percentage})")
    return results