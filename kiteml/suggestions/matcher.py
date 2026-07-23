"""
matcher.py — String similarity and fuzzy column matching algorithms for KiteML.
"""

from typing import Sequence


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def string_similarity(s1: str, s2: str) -> float:
    """
    Calculate normalized similarity between two strings (0.0 to 1.0).
    Combines exact match, case-insensitive match, token overlap, and Levenshtein similarity.
    """
    if not s1 or not s2:
        return 0.0

    s1_clean = s1.strip()
    s2_clean = s2.strip()

    # Exact match
    if s1_clean == s2_clean:
        return 1.0

    s1_lower = s1_clean.lower()
    s2_lower = s2_clean.lower()

    # Case-insensitive exact match
    if s1_lower == s2_lower:
        return 0.98

    # Levenshtein similarity
    max_len = max(len(s1_lower), len(s2_lower))
    dist = levenshtein_distance(s1_lower, s2_lower)
    lev_sim = 1.0 - (dist / max_len)

    # Prefix / Suffix bonus
    bonus = 0.0
    if s1_lower.startswith(s2_lower) or s2_lower.startswith(s1_lower):
        bonus += 0.1
    if s1_lower.endswith(s2_lower) or s2_lower.endswith(s1_lower):
        bonus += 0.1

    # Token overlap bonus
    t1 = set(s1_lower.replace("_", " ").replace("-", " ").split())
    t2 = set(s2_lower.replace("_", " ").replace("-", " ").split())
    if t1 and t2:
        intersection = t1.intersection(t2)
        if intersection:
            token_sim = len(intersection) / max(len(t1), len(t2))
            lev_sim = max(lev_sim, token_sim * 0.9)

    score = min(1.0, lev_sim + bonus)
    return round(score, 3)


def match_column_name(
    target_name: str,
    available_columns: Sequence[str],
    threshold: float = 0.4,
) -> list[tuple[str, float]]:
    """
    Find best matching column names for a given target string.

    Returns list of tuples (column_name, similarity_score) sorted descending by score.
    """
    if not target_name or not available_columns:
        return []

    results: list[tuple[str, float]] = []
    for col in available_columns:
        sim = string_similarity(target_name, col)
        if sim >= threshold:
            results.append((col, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results
