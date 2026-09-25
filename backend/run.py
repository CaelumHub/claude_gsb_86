"""
run.py
------
Entry point.  Start the server:

    python -m backend.run            # serve on 127.0.0.1:8080
    GSB_PORT=9000 python -m backend.run
    python backend/run.py --seed    # seed demo data on startup if empty

Also supports a ``--seed`` flag to auto-populate demo data when the dataset is
empty, and a ``--check`` flag to run a quick self-test of the algorithm stack
and exit (useful for CI / validation).
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running both as a package (`python -m backend.run`) and as a script
# (`python backend/run.py`).
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend import api, config, seed
    from backend.service import SocialGraphService
else:
    from . import api, config, seed
    from .service import SocialGraphService


def _check() -> int:
    """Run a smoke test over the algorithm stack; exit 0 on success."""
    from backend import algorithms
    from backend.graph import Graph

    g = Graph(directed=False)
    # A small barbell: two 3-cliques joined by a bridge.
    edges = [(1, 2), (2, 3), (1, 3), (3, 4), (4, 5), (5, 6), (4, 6)]
    for u, v in edges:
        g.add_edge(u, v)
    g.freeze()

    assert g.node_count == 6, g.node_count
    assert g.edge_count == 7, g.edge_count

    path, dist, alg = algorithms.shortest_path(g, 1, 6)
    assert path == [1, 2, 3, 4, 5, 6] or path == [1, 3, 4, 6, 5] or len(path) - 1 == dist, path

    common = algorithms.common_friends(g, 1, 6)
    pr = algorithms.pagerank(g)
    assert abs(sum(pr.values()) - 1.0) < 1e-6, sum(pr.values())

    lv = algorithms.louvain(g)
    assert lv["num_communities"] >= 2, lv  # cliques should separate

    rec = algorithms.hybrid_recommend(g, 1, k=3)
    assert "items" in rec

    # --- reachability: layered BFS fan-out -------------------------------
    comp = algorithms.bfs_component_layers(g, 1)
    assert comp["found"]
    assert [len(layer) for layer in comp["layers"]] == [1, 2, 1, 2], comp["layers"]
    assert comp["component_size"] == 6 and comp["eccentricity"] == 3

    view = algorithms.reachability_summary(comp, max_hops=2, total_nodes=6)
    assert [l["new_nodes"] for l in view["layers"]] == [1, 2, 1]
    assert [l["cumulative"] for l in view["layers"]] == [1, 3, 4]
    assert view["reachable_within_hops"] == 4
    assert view["unreachable_within_hops"] == 2
    assert view["truncated"] is True
    assert view["disconnected"] is False

    full_view = algorithms.reachability_summary(comp, max_hops=10, total_nodes=6)
    assert full_view["reachable_within_hops"] == 6
    assert full_view["truncated"] is False
    assert [l["cumulative"] for l in full_view["layers"]][-1] == 6

    # Disconnected graph: an extra isolated component must be flagged.
    g2 = Graph(directed=False)
    g2.add_edge(1, 2)
    g2.add_edge(7, 8)
    g2.freeze()
    comp2 = algorithms.bfs_component_layers(g2, 1)
    view2 = algorithms.reachability_summary(comp2, max_hops=5, total_nodes=4)
    assert view2["disconnected"] is True
    assert view2["reachable_total"] == 2
    assert view2["unreachable_total"] == 2
    assert algorithms.bfs_component_layers(g2, 99) == {"found": False}

    print("[check] OK: graph, bfs, pagerank, louvain, recommend, reachability all pass")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Social graph analysis server")
    parser.add_argument("--seed", action="store_true", help="seed demo data if empty")
    parser.add_argument("--check", action="store_true", help="run self-test and exit")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    if args.check:
        return _check()

    if args.host:
        config.HOST = args.host
    if args.port:
        config.PORT = args.port

    config.ensure_dirs()
    service = SocialGraphService()

    if args.seed:
        users = service.store.load_users()
        if not users:
            print("[social-graph] empty dataset -- seeding demo data ...")
            result = seed.generate_demo(service)
            print(f"[social-graph] seeded {result['users']} users, {result['edges']} edges")

    api.run(service)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
