from dataclasses import dataclass

@dataclass(frozen=True)
class AppConfig:
    top_alerts: int = 200
    temporal_window_days: int = 7
    graph_neighborhood_hops: int = 1
    random_state: int = 42
    max_betweenness_nodes: int = 1000
    max_graph_nodes = 200
