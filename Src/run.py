import time
import json
from tqdm import tqdm

from Src.positioning import position_entities
from Src.link import link_rerank_entity, link_topone
from Src.chain import create_entity_chain
from Src.stores.base import ConceptStore


def extract_and_position(text: str, llm) -> list:
    """Stages 1-2 (NER extraction + span positioning). Backend-independent."""
    ner_chain = create_entity_chain(llm=llm)
    response = ner_chain.invoke({"input_text": text})

    if not isinstance(response, dict) or "entities" not in response:
        raise ValueError(f"Invalid NER output: {response}")

    validated = position_entities(text, response["entities"])
    entities = []
    for ent in validated:
        if 'start_index' not in ent or 'end_index' not in ent:
            print(f"!!! Skipping entity '{ent.get('term', 'Unknown')}' because it could not be positioned in the text.")
            continue
        entities.append(ent)
    return entities


def predict_entities(
        text: str,
        store: ConceptStore,
        k_candidate: int,
        llm,
        reranker,
        use_hybrid: bool = False):
    entities = extract_and_position(text, llm)

    predictions = []
    for ent in entities:
        try:
            ranked = link_rerank_entity(
                entity=ent,
                store=store,
                k_candidate=k_candidate,
                llm=llm,
                reranker=reranker,
                use_hybrid=use_hybrid)
            if ranked is not None:
                predictions.append(ranked)
            else:
                print(f"!!! Skipped null reranked entity for '{ent['term']}'")
        except Exception as e:
            print(f"!!! Error reranking entity '{ent['term']}': {e}")
        time.sleep(5)

    return predictions



def run_batch(
        test_path: str,
        output_path: str,
        store: ConceptStore,
        k_candidate: int,
        llm,
        reranker,
        use_hybrid: bool = False):
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    all_predictions = []
    num_error = 0
    for item in tqdm(test_data):
        text = item["text"]
        try:
            preds = predict_entities(
                text,
                store=store,
                k_candidate=k_candidate,
                llm=llm,
                reranker=reranker,
                use_hybrid=use_hybrid)
            all_predictions.append(preds)
        except Exception as e:
            print(f"Error on id={item.get('id')}: {e}")
            all_predictions.append([])
            num_error += 1
        time.sleep(30)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=2)
    print(f"unable to process {num_error} documents")


def predict_top1(
        text: str,
        store: ConceptStore,
        k_candidate: int,
        llm,
        reranker,
        use_hybrid: bool = False):
    entities = extract_and_position(text, llm)

    predictions = []
    for ent in entities:
        try:
            ranked = link_topone(
                entity=ent,
                store=store,
                k_candidate=k_candidate,
                reranker=reranker,
                use_hybrid=use_hybrid)
            if ranked is not None:
                predictions.append(ranked)
            else:
                print(f"!!! Skipped null reranked entity for '{ent['term']}'")
        except Exception as e:
            print(f"!!! Error reranking entity '{ent['term']}': {e}")
        time.sleep(5)

    return predictions



def run_batch_top1(
        test_path: str,
        output_path: str,
        store: ConceptStore,
        k_candidate: int,
        llm,
        reranker,
        use_hybrid: bool = False):
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    all_predictions = []
    num_error = 0
    for item in tqdm(test_data):
        text = item["text"]
        try:
            preds = predict_top1(
                text,
                store=store,
                k_candidate=k_candidate,
                llm=llm,
                reranker=reranker,
                use_hybrid=use_hybrid)
            all_predictions.append(preds)
        except Exception as e:
            print(f"Error on id={item.get('id')}: {e}")
            all_predictions.append([])
            num_error += 1
        time.sleep(30)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=2)
    print(f"unable to process {num_error} documents")
