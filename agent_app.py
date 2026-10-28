import heapq
import itertools
import math
from collections import deque

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="DIT Campus Navigator", layout="wide")
st.title("🎓 DIT University - Intelligent Campus Navigation Agent")
st.caption(
    "CSN304 AI Project P-1 · BFS, DFS, UCS, Greedy and A* on a campus graph with "
    "obstacles, crowd levels and accessibility."
)

# ======================================================================
# CONSTANTS
# ======================================================================
WALK_SPEED = 80.0  # metres per minute
CROWD_LABEL = {0: "🟢 Low", 1: "🟡 Moderate", 2: "🔴 High"}
CROWD_TIME_FACTOR = {0: 1.0, 1: 1.25, 2: 1.6}  # walking slows down in crowds
CROWD_COLOR = {0: "#2ecc71", 1: "#f1c40f", 2: "#e74c3c"}
CROWD_PENALTY_M = 150  # metre-equivalents added per crowd level (Least crowded mode)

ALGOS = ["A*", "UCS (Dijkstra)", "Greedy Best-First", "BFS", "DFS"]
MODES = ["Shortest distance", "Fastest time (crowd-aware)", "Least crowded"]
UNIT = {MODES[0]: "m", MODES[1]: "min", MODES[2]: "m-eq"}

NODES = {
    "DIT Main Entrance": (9.0, 9.5),
    "Parking Area": (7.2, 9.0),
    "Sarojini Girls Hostel": (4.5, 8.2),
    "Vastu Building": (5.2, 6.8),
    "School of Designing": (6.0, 6.5),
    "Vedanta Computing Dept": (5.2, 5.2),
    "Vedanta CS Classroom 101": (4.2, 4.8),
    "Vedanta CS Lab 202": (6.2, 4.8),
    "Admin Block Chanakya": (3.2, 3.5),
    "Admin Office Room": (2.2, 3.2),
    "First Year Building Vishwakarma": (2.5, 2.2),
    "Vishwakarma Lecture Hall 1": (1.5, 1.8),
    "Bhabha Block": (1.2, 1.2),
    "Sarabhai Block": (2.2, 0.5),
    "Raman Block": (1.2, 0.4),
}

EDGES = [
    ("DIT Main Entrance", "Parking Area", 80, True),
    ("DIT Main Entrance", "Sarojini Girls Hostel", 220, True),
    ("Parking Area", "Vastu Building", 150, True),
    ("Vastu Building", "School of Designing", 60, True),
    ("Vastu Building", "Vedanta Computing Dept", 110, True),
    ("Vedanta Computing Dept", "Vedanta CS Classroom 101", 30, True),
    ("Vedanta Computing Dept", "Vedanta CS Lab 202", 35, True),
    ("Vedanta Computing Dept", "Admin Block Chanakya", 180, True),
    ("Admin Block Chanakya", "Admin Office Room", 25, True),
    ("Admin Block Chanakya", "First Year Building Vishwakarma", 120, True),
    ("First Year Building Vishwakarma", "Vishwakarma Lecture Hall 1", 30, True),
    ("First Year Building Vishwakarma", "Bhabha Block", 90, True),
    ("Bhabha Block", "Raman Block", 50, True),
    ("Raman Block", "Sarabhai Block", 80, True),
    ("Sarojini Girls Hostel", "Vastu Building", 100, False),
    ("Parking Area", "School of Designing", 140, True),
    ("School of Designing", "Vedanta CS Lab 202", 90, True),
    ("Vedanta CS Classroom 101", "Admin Block Chanakya", 75, False),
    ("Admin Block Chanakya", "Bhabha Block", 200, True),
    ("Sarabhai Block", "First Year Building Vishwakarma", 110, True),
]

CROWD_RULES = {
    "Morning Peak (09:00 AM - 10:00 AM)": {
        2: {"DIT Main Entrance", "Sarojini Girls Hostel"},
        1: {"Parking Area"},
    },
    "Class Hours (11:00 AM - 01:00 PM)": {
        1: {"Vedanta Computing Dept", "Admin Block Chanakya", "First Year Building Vishwakarma"},
    },
    "Lunch Break (01:00 PM - 02:00 PM)": {
        2: {"Vastu Building", "Vedanta Computing Dept"},
        1: {"School of Designing", "Vedanta CS Classroom 101", "Vedanta CS Lab 202"},
    },
    "Evening Rush (04:00 PM - 05:00 PM)": {
        2: {"DIT Main Entrance", "Parking Area"},
        1: {"Vastu Building"},
    },
    "Off-Peak / Quiet Hours": {},
}

def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

@st.cache_resource
def build_graph():
    G = nx.Graph()
    for name, p in NODES.items():
        G.add_node(name, pos=p)
    for u, v, w, acc in EDGES:
        G.add_edge(u, v, weight=w, accessible=acc)
    scale = min(d["weight"] / euclid(NODES[u], NODES[v]) for u, v, d in G.edges(data=True))
    return G, scale

G, SCALE = build_graph()
POS = nx.get_node_attributes(G, "pos")

def crowd_level(u, v, slot):
    lvl = 0
    for level, nodes in CROWD_RULES[slot].items():
        if u in nodes or v in nodes:
            lvl = max(lvl, level)
    return lvl

def make_cost(mode, slot):
    def cost(u, v, d):
        lvl = crowd_level(u, v, slot)
        if mode == MODES[0]:
            return d["weight"]
        if mode == MODES[1]:
            return d["weight"] / WALK_SPEED * CROWD_TIME_FACTOR[lvl]
        return d["weight"] + CROWD_PENALTY_M * lvl
    return cost

def make_h(goal, mode):
    def h(n):
        d = euclid(POS[n], POS[goal]) * SCALE
        return d / WALK_SPEED if mode == MODES[1] else d
    return h

def search(start, goal, algo, cost, h, blocked, accessible_only):
    informed = algo in ("A*", "UCS (Dijkstra)", "Greedy Best-First")
    frontier = [] if informed else deque()
    tick = itertools.count()
    closed, closed_set, parent, gcost, log = [], set(), {}, {}, []

    def neighbors(u):
        for v, d in G[u].items():
            if frozenset((u, v)) in blocked:
                continue
            if accessible_only and not d["accessible"]:
                continue
            yield v, d

    def push(node, par, g):
        if informed:
            hv = h(node)
            if algo == "UCS (Dijkstra)":
                f = g
            elif algo == "Greedy Best-First":
                f = hv
            else:
                f = g + hv
            heapq.heappush(frontier, (f, next(tick), node, par, g))
        else:
            frontier.append((node, par, g))

    def open_snapshot():
        out, seen = [], set()
        if informed:
            for f, _, n, _, _ in sorted(frontier):
                if n not in closed_set and n not in seen:
                    seen.add(n)
                    out.append(f"{n} (f={f:.1f})")
        else:
            items = list(frontier) if algo == "BFS" else list(reversed(frontier))
            for n, _, _ in items:
                if n not in closed_set and n not in seen:
                    seen.add(n)
                    out.append(n)
        return out

    push(start, None, 0.0)
    while frontier:
        if informed:
            _, _, node, par, g = heapq.heappop(frontier)
        elif algo == "BFS":
            node, par, g = frontier.popleft()
        else:
            node, par, g = frontier.pop()

        if node in closed_set:
            continue
        closed_set.add(node)
        closed.append(node)
        parent[node], gcost[node] = par, g

        if node != goal:
            for v, d in neighbors(node):
                if v not in closed_set:
                    push(v, node, g + cost(node, v, d))

        log.append(
            {
                "Step": len(log) + 1,
                "Expanded": node,
                "g(n)": round(g, 2),
                "Open list": " | ".join(open_snapshot()) or "-",
                "Closed list": ", ".join(closed),
            }
        )
        if node == goal:
            break

    path = []
    if goal in closed_set:
        n = goal
        while n is not None:
            path.append(n)
            n = parent[n]
        path.reverse()
    return {"path": path, "expanded": len(closed), "log": log}

def path_cost(path, cost):
    return sum(cost(path[i], path[i + 1], G[path[i]][path[i + 1]]) for i in range(len(path) - 1))

def path_stats(path, slot):
    dist = mins = 0.0
    worst = 0
    for u, v in zip(path, path[1:]):
        w = G[u][v]["weight"]
        lvl = crowd_level(u, v, slot)
        dist += w
        mins += w / WALK_SPEED * CROWD_TIME_FACTOR[lvl]
        worst = max(worst, lvl)
    return dist, mins, worst

# ======================================================================
# SIDEBAR
# ======================================================================
st.sidebar.header("🧭 Navigation & Search Controls")
locations = list(G.nodes())
slot = st.sidebar.selectbox("⏰ Campus Time Slot", list(CROWD_RULES))
start_node = st.sidebar.selectbox("Start Location", locations, index=0)
dest_node = st.sidebar.selectbox("Destination", locations, index=5)
algo = st.sidebar.selectbox("Search Algorithm", ALGOS)
mode = st.sidebar.selectbox("Route Preference", MODES)
accessible_only = st.sidebar.checkbox("♿ Accessible routes only (no stairs)")

edge_labels = {f"{u} ↔ {v}": frozenset((u, v)) for u, v in G.edges()}
blocked_choice = st.sidebar.multiselect("🚧 Blocked paths (obstacles)", list(edge_labels))
blocked = {edge_labels[b] for b in blocked_choice}
anim_speed = st.sidebar.slider("⚡ Animation speed (ms per frame)", 30, 150, 70, 10)

cost_fn = make_cost(mode, slot)
h_fn = make_h(dest_node, mode)

def run(a):
    return search(start_node, dest_node, a, cost_fn, h_fn, blocked, accessible_only)

result = run(algo)
path = result["path"]

def build_map():
    traces = []
    for lvl in (0, 1, 2):
        xs, ys = [], []
        for u, v in G.edges():
            if frozenset((u, v)) in blocked or crowd_level(u, v, slot) != lvl:
                continue
            xs += [POS[u][0], POS[v][0], None]
            ys += [POS[u][1], POS[v][1], None]
        traces.append(
            go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                       line=dict(width=2, color=CROWD_COLOR[lvl]),
                       name=f"{CROWD_LABEL[lvl]} crowd")
        )

    bx, by, mx, my = [], [], [], []
    for u, v in G.edges():
        if frozenset((u, v)) in blocked:
            bx += [POS[u][0], POS[v][0], None]
            by += [POS[u][1], POS[v][1], None]
            mx.append((POS[u][0] + POS[v][0]) / 2)
            my.append((POS[u][1] + POS[v][1]) / 2)
    traces.append(go.Scatter(x=bx, y=by, mode="lines", hoverinfo="skip",
                           line=dict(width=2, color="#888888", dash="dot"), name="Blocked"))
    traces.append(go.Scatter(x=mx, y=my, mode="text", text=["❌"] * len(mx), hoverinfo="skip",
                           showlegend=False))

    px, py = [], []
    for u, v in zip(path, path[1:]):
        px += [POS[u][0], POS[v][0], None]
        py += [POS[u][1], POS[v][1], None]
    traces.append(go.Scatter(x=px, y=py, mode="lines", hoverinfo="skip",
                           line=dict(width=7, color="#00FFFF"), name="Route"))

    colors = []
    for n in G.nodes():
        colors.append("#00FF66" if n == start_node else "#FF0055" if n == dest_node
                      else "#BF00FF" if n in path else "#1EAEDB")
    traces.append(go.Scatter(
        x=[POS[n][0] for n in G.nodes()], y=[POS[n][1] for n in G.nodes()],
        mode="markers+text", text=list(G.nodes()), textposition="top center",
        textfont=dict(size=10), hoverinfo="text", showlegend=False,
        marker=dict(color=colors, size=15, line=dict(width=2, color="#FFFFFF"))))

    coords = [POS[start_node]]
    for u, v in zip(path, path[1:]):
        (x0, y0), (x1, y1) = POS[u], POS[v]
        for t in np.linspace(0, 1, 10)[1:]:
            coords.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))

    agent_style = dict(size=24, color="#FFFF00", line=dict(width=3, color="#000000"))
    agent_idx = len(traces)
    traces.append(go.Scatter(x=[coords[0][0]], y=[coords[0][1]], mode="markers",
                           marker=agent_style, showlegend=False, hoverinfo="skip"))

    frames = [
        go.Frame(data=[go.Scatter(x=[x], y=[y], mode="markers", marker=agent_style)],
                 traces=[agent_idx], name=f"f{i}")
        for i, (x, y) in enumerate(coords)
    ]

    return go.Figure(
        data=traces,
        frames=frames,
        layout=go.Layout(
            title=dict(text=f"<b>{algo} · {mode} · {slot.split('(')[0].strip()}</b>",
                       font=dict(size=17)),
            template="plotly_dark", height=640, hovermode="closest",
            margin=dict(b=40, l=20, r=20, t=90),
            legend=dict(orientation="h", y=-0.02),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            updatemenus=[dict(
                type="buttons", showactive=False, direction="left", x=0.0, y=0.98, xanchor="left", yanchor="top",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, {"frame": {"duration": anim_speed, "redraw": False},
                                     "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="❚❚ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate", "transition": {"duration": 0}}]),
                ])],
        ),
    )

def show_plot(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)

st.subheader("🗺️ Live Interactive Campus Map")
show_plot(build_map())

st.markdown("---")
st.subheader(f"📊 Route Metrics ({slot})")

if path:
    dist, mins, worst = path_stats(path, slot)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📍 Distance", f"{dist:.0f} m")
    c2.metric("⏱️ Est. time (with crowd)", f"~{mins:.1f} min")
    c3.metric("👥 Worst crowd on route", CROWD_LABEL[worst])
    c4.metric(f"💰 Path cost ({UNIT[mode]})", f"{path_cost(path, cost_fn):.1f}")
    c5.metric("🔎 Nodes expanded", result["expanded"])
    st.markdown("**Final route:**")
    st.info(" ➔ ".join(path))
else:
    st.error("No valid route found - the destination is unreachable with the current blocks/accessibility setting.")

tab1, tab2, tab3 = st.tabs(["📋 Open / Closed lists", "⚖️ Algorithm comparison", "🧠 Problem formulation"])

with tab1:
    st.dataframe(pd.DataFrame(result["log"]), hide_index=True)

with tab2:
    rows = []
    for a in ALGOS:
        r = run(a)
        if r["path"]:
            d, m, _ = path_stats(r["path"], slot)
            rows.append({"Algorithm": a, "Nodes expanded": r["expanded"], "Hops": len(r["path"]) - 1,
                         "Distance (m)": round(d, 1), "Time (min)": round(m, 2),
                         f"Cost ({UNIT[mode]})": round(path_cost(r["path"], cost_fn), 2),
                         "Route": " ➔ ".join(r["path"])})
        else:
            rows.append({"Algorithm": a, "Nodes expanded": r["expanded"], "Route": "No route"})
    df = pd.DataFrame(rows)
    ccol = f"Cost ({UNIT[mode]})"
    if ccol in df and df[ccol].notna().any():
        best = df[ccol].min()
        df["Optimal?"] = df[ccol].apply(lambda c: "✅" if pd.notna(c) and c <= best + 1e-9 else "❌")
    st.dataframe(df, hide_index=True)

with tab3:
    st.markdown(
        f"""
**State:** the agent's current location (a node of the campus graph).
**Actions:** move along a non-blocked edge to a neighbouring location.
**Transition model:** result(location, move) = neighbour reached.
**Step cost:** distance (m), crowd-adjusted time (min) or crowd penalty, per *Route Preference*.
**Goal test:** location == destination.
**Heuristic (A*, Greedy):** straight-line map distance × {SCALE:.1f} m/unit. The scale is the smallest
metres-per-unit ratio over all edges, so h(n) never overestimates → admissible and consistent.
"""
    )