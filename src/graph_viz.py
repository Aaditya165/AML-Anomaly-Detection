from __future__ import annotations

from typing import Iterable, Sequence

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def build_risk_subgraph(df: pd.DataFrame, scored_accounts: pd.DataFrame, max_nodes: int = 150, hops: int = 1) -> nx.DiGraph:
    top_accounts = scored_accounts.sort_values("risk_score", ascending=False).head(max_nodes)["account"].astype(str).tolist()
    sub = df[df["from_account"].astype(str).isin(top_accounts) | df["to_account"].astype(str).isin(top_accounts)].copy()
    g = nx.DiGraph()
    for _, r in sub.iterrows():
        u = str(r["from_account"])
        v = str(r["to_account"])
        amt = float(r.get("amount_received", 0.0) or 0.0)
        g.add_edge(u, v, amount=amt, timestamp=r["timestamp"])
    # optionally expand by neighborhood around selected nodes
    if hops > 1 and len(g) > 0:
        expanded = set(top_accounts)
        frontier = set(top_accounts)
        for _ in range(hops):
            new = set()
            for n in frontier:
                new |= set(g.predecessors(n)) | set(g.successors(n))
            new -= expanded
            expanded |= new
            frontier = new
        g = g.subgraph(expanded).copy()
    return g

def plot_network(g: nx.DiGraph, scored_accounts: pd.DataFrame) -> go.Figure:
    if len(g) == 0:
        fig = go.Figure()
        fig.update_layout(
            title="High-risk subgraph",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    risk_map = scored_accounts.set_index("account")["risk_score"].to_dict()
    tier_map = scored_accounts.set_index("account")["risk_tier"].to_dict()

    pos = nx.spring_layout(g.to_undirected(), seed=42, k=None)

    edge_x = []
    edge_y = []
    edge_text = []
    for u, v, data in g.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(f"{u} → {v}: {data.get('amount', 0):,.2f}")

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.6),
        hoverinfo="none",
        mode="lines",
        name="transactions",
    )

    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    for n in g.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        score = float(risk_map.get(n, 0.0))
        tier = str(tier_map.get(n, "Low"))
        node_text.append(f"Account: {n}<br>Risk: {score:.3f}<br>Tier: {tier}")
        node_size.append(12 + 30 * score)
        node_color.append(score)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            showscale=True,
            colorscale="Viridis",
            line=dict(width=1, color="#222"),
            opacity=0.9,
        ),
        name="accounts",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="High-risk account network",
        showlegend=False,
        hovermode="closest",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template="plotly_white",
        height=700,
    )
    return fig
