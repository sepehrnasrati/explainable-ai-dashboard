from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Set


@dataclass(frozen=True)
class ECSResult:
    top_k: int
    shap_features: List[str]
    lime_features: List[str]
    intersection: List[str]
    union: List[str]
    ecs: float


def jaccard_similarity(a: Iterable[str], b: Iterable[str]) -> float:
    A: Set[str] = set(a)
    B: Set[str] = set(b)
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


def compute_ecs(shap_features: List[str], lime_features: List[str], top_k: int) -> ECSResult:
    shap_top = shap_features[:top_k]
    lime_top = lime_features[:top_k]

    A, B = set(shap_top), set(lime_top)
    inter = sorted(list(A & B))
    uni = sorted(list(A | B))
    ecs = jaccard_similarity(A, B)

    return ECSResult(
        top_k=top_k,
        shap_features=shap_top,
        lime_features=lime_top,
        intersection=inter,
        union=uni,
        ecs=ecs,
    )
