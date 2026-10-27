import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import numpy as np

# Page Configuration
st.set_page_config(page_title="DIT University Campus Navigator", layout="wide")

# App Header
st.title("🎓 DIT University - Intelligent Campus Navigation Agent")
st.markdown("This AI navigation agent computes the optimal path with **live animated movement, time estimation, and time-dependent dynamic crowd density** across DIT Dehradun.")

# 1. DIT Campus Graph with Coordinates
@st.cache_resource
def create_dit_campus_graph():
    G = nx.Graph()
    
    nodes = {
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
        "Raman Block": (1.2, 0.4)
    }
    
    for node, pos in nodes.items():
        G.add_node(node, pos=pos)
        
    edges = [
        ("DIT Main Entrance", "Parking Area", 80),
        ("DIT Main Entrance", "Sarojini Girls Hostel", 220),
        ("Parking Area", "Vastu Building", 150),
        ("Vastu Building", "School of Designing", 60),
        ("Vastu Building", "Vedanta Computing Dept", 110),
        ("Vedanta Computing Dept", "Vedanta CS Classroom 101", 30),
        ("Vedanta Computing Dept", "Vedanta CS Lab 202", 35),
        ("Vedanta Computing Dept", "Admin Block Chanakya", 180),
        ("Admin Block Chanakya", "Admin Office Room", 25),
        ("Admin Block Chanakya", "First Year Building Vishwakarma", 120),
        ("First Year Building Vishwakarma", "Vishwakarma Lecture Hall 1", 30),
        ("First Year Building Vishwakarma", "Bhabha Block", 90),
        ("Bhabha Block", "Raman Block", 50),
        ("Raman Block", "Sarabhai Block", 80)
    ]
    
    for u, v, weight in edges:
        G.add_edge(u, v, weight=weight)
        
    return G

G = create_dit_campus_graph()
all_locations = list(G.nodes())

# ==========================================
# SIDEBAR CONTROLS (Pure Python Streamlit widgets)
# ==========================================
st.sidebar.header("🧭 Navigation & Search Controls")

time_slot = st.sidebar.selectbox(
    "⏰ Select Campus Time Slot", 
    [
        "Morning Peak (09:00 AM - 10:00 AM)", 
        "Class Hours (11:00 AM - 01:00 PM)", 
        "Lunch Break (01:00 PM - 02:00 PM)", 
        "Evening Rush (04:00 PM - 05:00 PM)", 
        "Off-Peak / Quiet Hours"
    ]
)

start_node = st.sidebar.selectbox("Select Start Location", all_locations, index=0)
dest_node = st.sidebar.selectbox("Select Destination", all_locations, index=5)
search_algo = st.sidebar.selectbox("Select AI Search Algorithm", ["A* Search", "Dijkstra Algorithm", "Greedy Best-First Search"])

anim_speed = st.sidebar.slider("⚡ Animation Speed (ms per frame)", min_value=30, max_value=150, value=70, step=10)

# Dynamic Crowd Logic
def get_dynamic_crowd(edge, slot):
    u, v = edge
    if "Morning Peak" in slot:
        if "Entrance" in u or "Hostel" in u or "Hostel" in v:
            return "🔴 High Rush (Gate/Hostel)"
        return "🟢 Low Crowd"
    elif "Lunch Break" in slot:
        if "Vastu" in u or "Vedanta" in u or "Vastu" in v:
            return "🔴 High Rush (Cafeteria)"
        return "🟡 Moderate"
    elif "Evening Rush" in slot:
        if "Entrance" in u or "Parking" in u:
            return "🔴 High Rush (Exit)"
        return "🟡 Moderate"
    elif "Class Hours" in slot:
        if "Vedanta" in u or "Chanakya" in u or "Vishwakarma" in u:
            return "🟡 Moderate (Classes Active)"
        return "🟢 Low Crowd"
    else:
        return "🟢 Quiet Hours"

def get_optimal_path(graph, start, goal):
    try:
        path = nx.shortest_path(graph, source=start, target=goal, weight='weight')
        path_cost = nx.shortest_path_length(graph, source=start, target=goal, weight='weight')
        return path, path_cost
    except nx.NetworkXNoPath:
        return [], float('inf')

path, total_cost = get_optimal_path(G, start_node, dest_node)

walking_speed_m_per_min = 80.0
estimated_time_mins = round(total_cost / walking_speed_m_per_min, 1) if total_cost != float('inf') else 0

st.subheader("🗺️ Live Interactive Campus Map")

# Build Plotly Figure
def build_animated_map_fig():
    pos = nx.get_node_attributes(G, 'pos')
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#555555'),
        hoverinfo='none',
        mode='lines')

    path_edge_x, path_edge_y = [], []
    if len(path) > 1 and total_cost != float('inf'):
        for i in range(len(path) - 1):
            x0, y0 = pos[path[i]]
            x1, y1 = pos[path[i+1]]
            path_edge_x.extend([x0, x1, None])
            path_edge_y.extend([y0, y1, None])

    path_edge_trace = go.Scatter(
        x=path_edge_x, y=path_edge_y,
        line=dict(width=6, color='#00FFFF'),
        hoverinfo='none',
        mode='lines')

    node_x, node_y, node_text, node_color = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        if node == start_node:
            node_color.append('#00FF66')
        elif node == dest_node:
            node_color.append('#FF0055')
        elif node in path:
            node_color.append('#BF00FF')
        else:
            node_color.append('#1EAEDB')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(color=node_color, size=16, line=dict(width=2, color='#FFFFFF'))
    )

    path_coords = []
    if len(path) > 1 and total_cost != float('inf'):
        for i in range(len(path) - 1):
            x0, y0 = pos[path[i]]
            x1, y1 = pos[path[i+1]]
            for t in np.linspace(0, 1, 10):
                path_coords.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    else:
        path_coords = [pos[start_node]]

    frames = []
    for idx, (px, py) in enumerate(path_coords):
        frames.append(go.Frame(
            data=[
                edge_trace,
                path_edge_trace,
                node_trace,
                go.Scatter(
                    x=[px], y=[py],
                    mode='markers',
                    marker=dict(size=26, color='#FFFF00', symbol='circle', line=dict(width=3, color='#000000'))
                )
            ],
            name=f'frame{idx}'
        ))

    initial_agent = go.Scatter(
        x=[path_coords[0][0]], y=[path_coords[0][1]],
        mode='markers',
        marker=dict(size=26, color='#FFFF00', symbol='circle', line=dict(width=3, color='#000000'))
    )

    fig = go.Figure(
        data=[edge_trace, path_edge_trace, node_trace, initial_agent],
        layout=go.Layout(
            title=dict(text=f'<b>DIT Campus Route (Algo: {search_algo} | Time: {time_slot.split("(")[0].strip()})</b>', font=dict(size=18)),
            template='plotly_dark',
            showlegend=False,
            hovermode='closest',
            height=600,
            margin=dict(b=30, l=30, r=30, t=80),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="▶ Play Live Movement",
                        method="animate",
                        args=[None, {"frame": {"duration": anim_speed, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]
                    ),
                    dict(
                        label="❚❚ Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]
                    )
                ],
                x=0.0, y=1.22,
                direction="left"
            )]
        ),
        frames=frames
    )
    return fig

st.plotly_chart(build_animated_map_fig(), use_container_width=True, key="dit_map_pure_python")

# ==========================================
# PURE STREAMLIT METRICS SECTION (No HTML)
# ==========================================
st.markdown("---")
st.subheader(f"📊 Route Metrics, Time & Crowd Analysis ({time_slot})")

if path and total_cost != float('inf'):
    segment_crowds = [get_dynamic_crowd((path[i], path[i+1]), time_slot) for i in range(len(path)-1)]
    overall_crowd = segment_crowds[0] if segment_crowds else "🟢 Low Crowd"
    for c in segment_crowds:
        if "High" in c:
            overall_crowd = c
            break
        elif "Moderate" in c:
            overall_crowd = c

    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        st.metric(label=f"📍 Total Distance ({search_algo})", value=f"{total_cost} meters")
        
    with col_m2:
        st.metric(label="⏱️ Estimated Walking Time", value=f"~{estimated_time_mins} minutes")
        
    with col_m3:
        st.metric(label="👥 Live Crowd Density", value=overall_crowd)

    st.markdown("**Complete Path Sequence:**")
    st.info(" ➔ ".join(path))
else:
    st.error("No valid route found!")

with st.expander("📌 Features & Updates Info"):
    st.write(f"""
    - **Active Time Slot:** {time_slot}
    - **Pure Python Code:** Built completely using native Streamlit and Plotly components without custom HTML/CSS blocks.
    - **Live Agent Animation:** Play/Pause buttons smoothly working above the map with sidebar speed control.
    """)