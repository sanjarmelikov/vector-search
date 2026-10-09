import numpy as np
import pytest

from vector.index import FlatIndex, HNSWIndex

# The 6-point map from Lessons 1–2 (notes/STUDY_GUIDE.md, Phase 3).
POINTS = {"A": (0, 0), "B": (2, 0), "C": (4, 0), "D": (3, 1), "E": (6, 2), "F": (8, 3)}
NAMES = list(POINTS)
Q = (7, 3)


def lesson_index(roads: dict[str, str]) -> HNSWIndex:
    """A one-layer index wired by hand to the lesson's map.

    The index scores by dot product (higher = closer) but the lesson uses
    distance (lower = closer). Writing each point p as (2p, -|p|²) and the
    query q as (q, 1) makes  p·q = |q|² - |p - q|²,  so a higher dot product
    means exactly a smaller distance, and the walk is the same as on paper.
    """
    index = HNSWIndex(dim=3)
    index._vectors = np.array(
        [(2 * x, 2 * y, -(x * x + y * y)) for x, y in POINTS.values()], dtype=np.float32
    )
    index._n = len(POINTS)
    index._layers = [{i: [NAMES.index(n) for n in roads[name]] for i, name in enumerate(NAMES)}]
    index._levels = [0] * len(POINTS)
    return index


def walk(index: HNSWIndex, start: str, ef: int) -> str:
    query = np.array([Q[0], Q[1], 1], dtype=np.float32)
    best = index._search_layer(query, [NAMES.index(start)], ef, layer=0)
    return NAMES[best[0][1]]


FULL = {"A": "BD", "B": "AC", "C": "BD", "D": "ACE", "E": "DF", "F": "E"}
BROKEN = {"A": "BD", "B": "AC", "C": "BD", "D": "AC", "E": "F", "F": "E"}  # no D–E road


def test_lesson1_greedy_walk_finds_F():
    assert walk(lesson_index(FULL), "A", ef=1) == "F"


def test_lesson1b_greedy_walk_gets_stuck_at_C_without_the_D_E_road():
    assert walk(lesson_index(BROKEN), "A", ef=1) == "C"
    assert walk(lesson_index(BROKEN), "E", ef=1) == "F"  # where you start matters


def test_lesson2_wider_beam_escapes_the_dead_end():
    # Add a B–E road. Greedy still stops at C (B looks worse than C), but a beam
    # of 3 keeps B alive long enough to explore it and reach E, then F.
    roads = dict(BROKEN, B="ACE", E="BF")
    assert walk(lesson_index(roads), "A", ef=1) == "C"
    assert walk(lesson_index(roads), "A", ef=3) == "F"


def random_unit_vectors(n, dim, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim)).astype(np.float32)


def recall_at_k(approx: HNSWIndex, exact: FlatIndex, queries: np.ndarray, k: int) -> float:
    hits = 0
    for q in queries:
        truth = set(exact.search(q, k)[0].tolist())
        hits += len(truth & set(approx.search(q, k)[0].tolist()))
    return hits / (k * len(queries))


def test_recall_against_exact_search():
    data = random_unit_vectors(2000, 32, seed=0)
    queries = random_unit_vectors(50, 32, seed=1)
    exact = FlatIndex(32)
    exact.add(data)
    approx = HNSWIndex(32, M=16, ef_construction=100, ef_search=64)
    approx.add(data[:1000])
    approx.add(data[1000:])  # ids continue across add() calls, as in FlatIndex
    assert len(approx) == 2000
    assert recall_at_k(approx, exact, queries, k=10) >= 0.95


def test_scores_match_exact_cosine_and_come_best_first():
    data = random_unit_vectors(300, 16, seed=2)
    index = HNSWIndex(16, M=8)
    index.add(data)
    q = random_unit_vectors(1, 16, seed=3)[0]
    ids, scores = index.search(q, k=5)
    cos = data @ q / (np.linalg.norm(data, axis=1) * np.linalg.norm(q))
    assert np.allclose(scores, cos[ids], atol=1e-5)
    assert np.all(np.diff(scores) <= 0)


def test_search_compares_against_a_fraction_of_the_index():
    data = random_unit_vectors(5000, 16, seed=4)
    index = HNSWIndex(16, M=8, ef_construction=64, ef_search=16)
    index.add(data)
    index.search(data[0], k=10)
    assert 0 < index.scored_last_search < len(data) / 2


def test_links_respect_the_limits():
    index = HNSWIndex(8, M=4, ef_construction=32)
    index.add(random_unit_vectors(500, 8, seed=5))
    for layer, graph in enumerate(index._layers):
        limit = index.M0 if layer == 0 else index.M
        assert all(len(links) <= limit for links in graph.values())
        assert all(node not in links for node, links in graph.items())  # no self-loops
    assert len(index._layers[0]) == 500  # layer 0 holds every node


def test_same_seed_same_graph():
    data = random_unit_vectors(200, 8, seed=6)
    a, b = HNSWIndex(8, M=4, seed=7), HNSWIndex(8, M=4, seed=7)
    a.add(data)
    b.add(data)
    assert a._layers == b._layers


def test_empty_index_small_index_and_wrong_dimension():
    index = HNSWIndex(4)
    assert len(index.search(np.ones(4), k=5)[0]) == 0
    index.add(np.eye(4))
    ids, _ = index.search(np.array([1, 0, 0, 0]), k=10)
    assert sorted(ids.tolist()) == [0, 1, 2, 3] and ids[0] == 0
    with pytest.raises(ValueError):
        index.add(np.ones((2, 3)))
    with pytest.raises(ValueError):
        HNSWIndex(4, M=1)
