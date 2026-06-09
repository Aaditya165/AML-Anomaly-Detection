from dataclasses import dataclass

@dataclass(frozen=True)
class AppConfig:
    max_graph_nodes: int = 150
    top_alerts: int = 200
    temporal_window_days: int = 7
    graph_neighborhood_hops: int = 1
    random_state: int = 42
    suspicious_threshold_high: float = 0.75
    suspicious_threshold_medium: float = 0.50
    max_betweenness_nodes: int = 1000
