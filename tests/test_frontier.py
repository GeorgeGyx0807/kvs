from src.frontier import FrontierPoint, pareto_frontier


def test_pareto_frontier_removes_dominated_points():
    points = [
        FrontierPoint("a", 0.1, 0.8, 10, 5),
        FrontierPoint("b", 0.1, 0.9, 12, 6),
        FrontierPoint("c", 0.1, 0.85, 9, 4),
    ]
    frontier = pareto_frontier(points)
    assert {p.selector for p in frontier} == {"b", "c"}

