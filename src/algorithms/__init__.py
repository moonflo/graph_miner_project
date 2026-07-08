"""Classical graph algorithms used by the project."""

from .evaluation import (
    evaluate_ogb_style,
    hits_at_k_from_scores,
    mrr_from_citation2_scores,
)
from .link_prediction import (
    LinkPredictionScore,
    SUPPORTED_LINK_PREDICTION_METHODS,
    score_candidate_pairs,
    score_candidate_pairs_in_order,
    to_simple_undirected_for_topology,
)
from .scoring import (
    DatasetCandidateScores,
    OfficialOGBResult,
    score_ogb_official_for_dataset,
    score_ogb_official_multiple_methods,
    score_candidates_for_dataset,
    score_multiple_methods_for_dataset,
)

__all__ = [
    "DatasetCandidateScores",
    "LinkPredictionScore",
    "OfficialOGBResult",
    "SUPPORTED_LINK_PREDICTION_METHODS",
    "evaluate_ogb_style",
    "hits_at_k_from_scores",
    "mrr_from_citation2_scores",
    "score_candidate_pairs",
    "score_candidate_pairs_in_order",
    "score_candidates_for_dataset",
    "score_multiple_methods_for_dataset",
    "score_ogb_official_for_dataset",
    "score_ogb_official_multiple_methods",
    "to_simple_undirected_for_topology",
]
