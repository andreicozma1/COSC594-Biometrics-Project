"""Single-query Market-1501 ranking, CMC, and average precision.

The AP calculation follows the pinned Torchreid dependency's Python evaluator.
Metrics are fractions in [0, 1]; formatting as percentages belongs
to the output layer.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torchreid.metrics import compute_distance_matrix

from .configuration import DEFAULT_BATCH_SIZE, DEFAULT_DISTANCE, DistanceMetric
from .data import ImageRecord


# Torchreid calls its squared Euclidean implementation "euclidean".
_TORCHREID_METRIC = {
    DistanceMetric.SQUARED_EUCLIDEAN: "euclidean",
    DistanceMetric.COSINE: "cosine",
}


@dataclass(frozen=True)
class QueryResult:
    """Store the score and first correct rank for one query.

    AP and first_match_rank are None when filtering leaves no positive match.
    Ranks are one-based; AP is a fraction.
    """

    filename: str
    pid: int
    camera: int
    ap: float | None
    first_match_rank: int | None


@dataclass(frozen=True)
class EvaluationMetrics:
    """Summarize retrieval performance and the evaluated population.

    Rank-1/5/10 and mAP are fractions over valid_queries. Skipped queries are
    counted separately and excluded from those denominators.
    """

    mAP: float
    rank1: float
    rank5: float
    rank10: float
    total_queries: int
    valid_queries: int
    skipped_queries: int


@dataclass(frozen=True)
class Evaluation:
    """Pair aggregate metrics with results for every input query.

    The queries list preserves input order, including queries that could not
    be scored. Distance records how those rankings were calculated.
    """

    metrics: EvaluationMetrics
    queries: list[QueryResult]
    distance: DistanceMetric = DEFAULT_DISTANCE


def validate_features(features: np.ndarray, count: int) -> None:
    """Check that a matrix contains one finite vector per image.

    Args:
        features: Two-dimensional floating-point array with at least one column.
        count: Required number of rows.

    Raises:
        ValueError: Dimensions, numeric type, or values are invalid.
    """
    if features.ndim != 2 or features.shape[0] != count or features.shape[1] == 0:
        raise ValueError(f"Expected {count} embedding rows, got {features.shape}")
    if not np.issubdtype(features.dtype, np.floating) or not np.isfinite(features).all():
        raise ValueError("Embeddings must be finite floating-point values.")


def squared_distances(queries: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Return squared Euclidean distances between two embedding matrices.

    Args:
        queries: Finite floating-point [N, D] matrix.
        gallery: Finite floating-point [M, D] matrix with the same width.

    Returns:
        Float32 [N, M] distances. Inputs are not normalized or modified.

    Torchreid names this calculation ``euclidean`` even though it omits the
    square root. The caller limits query rows to bound temporary memory.
    """
    return pairwise_distances(queries, gallery, DistanceMetric.SQUARED_EUCLIDEAN)


def pairwise_distances(
    queries: np.ndarray,
    gallery: np.ndarray,
    metric: DistanceMetric = DEFAULT_DISTANCE,
) -> np.ndarray:
    """Compare every query embedding with every gallery embedding.

    Args:
        queries: Finite floating-point [N, D] matrix.
        gallery: Finite floating-point [M, D] matrix with the same width.
        metric: Squared Euclidean or cosine distance.

    Returns:
        Float32 [N, M] distances, where smaller values are better matches.

    The calculation follows Torchreid's evaluation implementation. Its cosine
    path applies L2 normalization with a small epsilon before matrix multiplication.
    """
    validate_features(queries, len(queries))
    validate_features(gallery, len(gallery))
    if queries.shape[1] != gallery.shape[1]:
        raise ValueError("Query and gallery embedding dimensions differ.")

    return _distance_matrix(
        torch.as_tensor(queries, dtype=torch.float32),
        torch.as_tensor(gallery, dtype=torch.float32),
        DistanceMetric(metric),
    )


def _distance_matrix(
    queries: torch.Tensor,
    gallery: torch.Tensor,
    metric: DistanceMetric,
) -> np.ndarray:
    """Run Torchreid's distance calculation for one query block.

    Returns:
        A float32 [query_count, gallery_count] NumPy array. Nonfinite arithmetic
        raises ValueError before ranking.
    """
    result = compute_distance_matrix(
        queries,
        gallery,
        metric=_TORCHREID_METRIC[metric],
    ).numpy()

    if not np.isfinite(result).all():
        raise ValueError("Distance calculation produced nonfinite values.")
    return result


def eligible_gallery_indices(query: ImageRecord, gallery: Sequence[ImageRecord]) -> np.ndarray:
    """Select gallery entries allowed by the Market-1501 protocol.

    Returns:
        Indices in original gallery order after excluding junk and entries
        sharing both the query identity and camera. Identity 0 is retained.
    """
    return np.array(
        [
            i
            for i, item in enumerate(gallery)
            if item.pid != -1 and not (item.pid == query.pid and item.camera == query.camera)
        ],
        dtype=np.int64,
    )


def rank_gallery(
    query: ImageRecord,
    gallery: Sequence[ImageRecord],
    distances: np.ndarray,
) -> np.ndarray:
    """Rank gallery images using the Market-1501 exclusions.

    Args:
        query: Query identity and camera labels.
        gallery: Image records aligned with the supplied distance vector.
        distances: One finite distance per gallery record.

    Returns:
        Gallery indices in increasing distance order, with ties resolved by
        filename. Junk and same-person/same-camera entries are excluded;
        other identities in the query camera and distractors remain.
    """
    if distances.shape != (len(gallery),) or not np.isfinite(distances).all():
        raise ValueError("Expected one finite distance per gallery image.")
    indices = eligible_gallery_indices(query, gallery)
    filenames = np.array([gallery[i].filename for i in indices])
    return indices[np.lexsort((filenames, distances[indices]))]


def average_precision(positive_ranks: np.ndarray) -> float | None:
    """Compute average precision from the ranks of correct matches.

    Args:
        positive_ranks: Increasing one-based ranks in the full filtered gallery.

    Returns:
        Precision averaged at each correct match, or None for an empty array.
    """
    if not len(positive_ranks):
        return None
    return float(np.mean(np.arange(1, len(positive_ranks) + 1) / positive_ranks))


def evaluate(
    queries: Sequence[ImageRecord],
    gallery: Sequence[ImageRecord],
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    batch_size: int = DEFAULT_BATCH_SIZE,
    distance: DistanceMetric = DEFAULT_DISTANCE,
) -> Evaluation:
    """Evaluate query embeddings against the full filtered gallery.

    Args:
        queries: Nonempty query records with positive identity labels.
        gallery: Nonempty gallery records in embedding row order.
        query_features: Floating-point [query_count, D] embedding matrix.
        gallery_features: Floating-point [gallery_count, D] embedding matrix.
        batch_size: Maximum queries per distance block; must be positive.
        distance: Measure used to rank each query against the gallery.

    Returns:
        Aggregate metrics and a result for every query. Queries with no eligible
        positive match have AP=None and are excluded from aggregate denominators.

    Raises:
        ValueError: Labels or embedding matrices are invalid, or no query can
            be evaluated.
    """
    if not queries or not gallery or batch_size < 1:
        raise ValueError("Evaluation requires nonempty splits and a positive batch size.")
    if any(query.pid <= 0 for query in queries):
        raise ValueError("Queries must have positive person identities.")
    validate_features(query_features, len(queries))
    validate_features(gallery_features, len(gallery))
    if query_features.shape[1] != gallery_features.shape[1]:
        raise ValueError("Query and gallery embedding dimensions differ.")

    distance = DistanceMetric(distance)

    # The same tensor serves every query batch, avoiding repeated gallery copies.
    gallery_tensor = torch.as_tensor(gallery_features, dtype=torch.float32)
    results: list[QueryResult] = []
    for start in range(0, len(queries), batch_size):
        query_tensor = torch.as_tensor(
            query_features[start : start + batch_size],
            dtype=torch.float32,
        )
        block = _distance_matrix(
            query_tensor,
            gallery_tensor,
            distance,
        )
        for offset, distances in enumerate(block):
            query = queries[start + offset]
            results.append(_evaluate_query(query, gallery, distances))

    return Evaluation(_aggregate_metrics(results), results, distance)


def _evaluate_query(
    query: ImageRecord,
    gallery: Sequence[ImageRecord],
    distances: np.ndarray,
) -> QueryResult:
    """Score one query using its full filtered ranking.

    Returns:
        AP and the first correct rank, with both values absent when the gallery
        has no eligible image of the query identity.
    """
    order = rank_gallery(query, gallery, distances)
    matches = np.array([gallery[index].pid == query.pid for index in order], dtype=bool)
    positive_ranks = np.flatnonzero(matches) + 1
    first_match = int(positive_ranks[0]) if len(positive_ranks) else None

    return QueryResult(
        filename=query.filename,
        pid=query.pid,
        camera=query.camera,
        ap=average_precision(positive_ranks),
        first_match_rank=first_match,
    )


def _aggregate_metrics(results: Sequence[QueryResult]) -> EvaluationMetrics:
    """Calculate mean AP and CMC using only scorable queries.

    Returns:
        Metric fractions together with total, valid, and skipped query counts.

    Raises:
        ValueError: None of the queries has a usable positive match.
    """
    average_precisions = [result.ap for result in results if result.ap is not None]
    if not average_precisions:
        raise ValueError("No query has an eligible positive gallery match after filtering.")

    first_matches = [
        result.first_match_rank
        for result in results
        if result.ap is not None and result.first_match_rank is not None
    ]
    valid_count = len(average_precisions)
    cmc = {rank: sum(first <= rank for first in first_matches) / valid_count for rank in (1, 5, 10)}

    return EvaluationMetrics(
        mAP=float(np.mean(average_precisions)),
        rank1=cmc[1],
        rank5=cmc[5],
        rank10=cmc[10],
        total_queries=len(results),
        valid_queries=valid_count,
        skipped_queries=len(results) - valid_count,
    )
