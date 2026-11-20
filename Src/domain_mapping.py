from typing import Dict, List

TYPE_TO_SEMANTIC_TAG: Dict[str, List[str]] = {
    "findings":    ["finding", "disorder", "situation", "morphologic abnormality"],
    "procedure":   ["procedure", "regime/therapy", "situation"],
    "medication":  ["substance", "product", "medicinal product", "clinical drug", "organism"],
    "laboratory":  ["procedure", "substance"],
    "anatomy":     ["body structure", "cell structure", "morphologic abnormality"],
    "event":       ["event", "physical force"],
    "observable":  ["observable entity", "record artifact", "organism", "physical object"],
    "score":       ["staging scale", "assessment scale", "observable entity"],
}


def allowed_tags_for_domain(domain: str) -> set:
    return {t.lower() for t in TYPE_TO_SEMANTIC_TAG.get(domain, [])}
