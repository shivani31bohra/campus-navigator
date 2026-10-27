import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import heapq

st.set_page_config(page_title="Intelligent Campus Navigation Agent", layout="wide")

st.title("🎓 Intelligent Campus Navigation Agent (AI Project)")
st.markdown("Yeh agent university ke **Classrooms, Laboratories, Offices, aur Departments** ke beech optimal route find karta hai.")

# Task 5: Environment Classification Display
with st.expander("📋 Task 5: Environment Classification Details"):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observability", "Fully Observable", "Complete campus map access")
    c2.metric("Determinism", "Deterministic", "Predictable actions & costs")
    c3.metric("Horizon", "Sequential", "Current decisions impact future path")
    c4.metric("Dynamics", "Dynamic", "Real-time obstacles & crowd changes")

# 1. Campus Graph Representation (Classrooms, Labs, Offices, Departments)
@st.cache_resource
def create_campus_graph():
    G = nx.Graph()
    
    # Nodes with categories (Classrooms, Labs, Offices, Departments, Hostels)
    nodes_data = {
        "Main Entrance": {"pos": (0, 0), "type": "Entrance"},
        "Admin Office": {"pos": (2, -1), "type": "Office"},
        "Vedanta Computing Dept": {"pos": (2, 2), "type": "Department"},
        "AI & ML Laboratory": {"pos": (1, 3.5), "type": "Laboratory"},
        "CSE Classroom 101": {"pos": (3, 3.5), "type": "Classroom"},
        "ECE Department": {"pos": (4, 0), "type": "Department"},
        "IoT & Embedded Lab": {"pos": (5, 1.5), "type": "Laboratory"},
        "Central Library": {"pos": (3, 1), "type": "Facility"},
        "Sarojini Girls Hostel": {"pos": (0.5, 5), "type": "Hostel"},
        "Sarabhai Boys Hostel": {"pos": (6, -3), "type": "Hostel"},
        "Campus Cafeteria": {"pos": (3, -2.5), "type": "Facility"}
    }
    
    for node, data in nodes_data.items():
        G.add_node(node, pos=data["pos"], type=data["type"])
        
    # Edges with metrics: distance (meters), time (mins), accessibility (1-5), crowd (1-10)
    edges = [
        ("Main Entrance", "Admin Office", {"distance": 120, "time": 2, "accessibility": 5, "crowd": 4}),
        ("Main Entrance", "Vedanta Computing Dept", {"distance": 180, "time": 3, "accessibility": 4, "crowd": 8}),
        ("Admin Office", "Central Library", {"distance": 90, "time": 2, "accessibility": 5, "crowd": 3}),
        ("Vedanta Computing Dept", "AI & ML Laboratory", {"distance": 50, "time": 1, "accessibility": 5, "crowd": 6}),
        ("Vedanta Computing Dept", "CSE Classroom 101", {"distance": 60, "time": 1, "accessibility": 4, "crowd": 9}),
        ("AI & ML Laboratory", "Sarojini Girls Hostel", {"distance": 110, "time": 2, "accessibility": 3, "crowd": 5}),
        ("Central Library", "ECE Department", {"distance": 140, "time": 3, "accessibility": 4, "crowd": 4}),
        ("ECE Department", "IoT & Embedded Lab", {"distance": 70, "time": 1, "accessibility": 5, "crowd": 5}),
        ("Admin Office", "Campus Cafeteria", {"distance": 150, "time": 3, "accessibility": 4, "crowd": 7}),
        ("Campus Cafeteria", "Sarabhai Boys Hostel", {"distance": 130, "time": 2, "accessibility": 4, "crowd": 6}),
        ("ECE Department", "Sarabhai Boys Hostel", {"distance": 200, "time": 4, "accessibility": 3, "crowd": 3})
    ]
    
    for u, v, data in edges:
        G.add_edge(u, v, **data)
        
    return G

G = create_campus_graph()

# Sidebar Navigation Controls
st.sidebar.header("🧭 Navigation & Search Controls")
all_nodes = list(G.nodes())

start_node = st.sidebar.selectbox("Select Start Location (State)", all_nodes, index=0)
goal_node = st.sidebar.selectbox("Select Destination (State)", all_nodes, index=3)

algorithm = st.sidebar.selectbox("Select AI Search Algorithm", ["A* Search", "Breadth-First Search (BFS)", "Depth-First Search (DFS)"])
optimization_metric = st.sidebar.selectbox("Advanced Route Selection", [
    "Distance (meters)", 
    "Estimated Time (mins)", 
    "Accessibility (Wheelchair friendly)", 
    "Crowd Level (Avoid Crowds)"
])

st.sidebar.subheader("🚫 Obstacle & Construction Simulation (Task 6)")
blocked_edges = st.sidebar.multiselect("Select Blocked Paths", list(G.edges()))

# Working graph after removing blocked paths
G_working = G.copy()
for u, v in blocked_edges:
    if G_working.has_edge(u, v):
        G_working.remove_edge(u, v)

# Heuristic function for A*
def heuristic(n1, n2):
    p1 = G.nodes[n1]['pos']
    p2 = G.nodes[n2]['pos']
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5 * 50

metric_map = {
    "Distance (meters)": "distance",
    "Estimated Time (mins)": "time",
    "Accessibility (Wheelchair friendly)": "accessibility",
    "Crowd Level (Avoid Crowds)": "crowd"
}
weight_attr = metric_map[optimization_metric]

# Search Algorithms with Open/Closed List Tracking (Task 3, 4, 7)
def run_bfs(graph, start, goal):
    queue = [[start]]
    visited = {start}
    closed_list = []
    open_snapshots = []

    while queue:
        path = queue.pop(0)
        node = path[-1]
        closed_list.append(node)
        open_snapshots.append(list(visited))

        if node == goal:
            return path, closed_list, open_snapshots

        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
    return None, closed_list, open_snapshots

def run_dfs(graph, start, goal):
    stack = [[start]]
    visited = set()
    closed_list = []
    open_snapshots = []

    while stack:
        path = stack.pop()
        node = path[-1]
        
        if node not in visited:
            visited.add(node)
            closed_list.append(node)
            open_snapshots.append(list(visited))

            if node == goal:
                return path, closed_list, open_snapshots

            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    new_path = list(path)
                    new_path.append(neighbor)
                    stack.append(new_path)
    return None, closed_list, open_snapshots

def run_astar(graph, start, goal, weight):
    pq = [(0, 0, start, [start])]
    g_costs = {start: 0}
    closed_list = []
    open_snapshots = []
    
    while pq:
        f, g, current, path = heapq.heappop(pq)
        
        if current in closed_list:
            continue
        closed_list.append(current)
        open_snapshots.append(list(g_costs.keys()))
        
        if current == goal:
            return path, closed_list, open_snapshots
            
        for neighbor in graph.neighbors(current):
            edge_weight = graph[current][neighbor].get(weight, 1)
            if weight == "accessibility":
                edge_weight = 6 - edge_weight # invert scale for optimization
                
            new_g = g + edge_weight
            
            if neighbor not in g_costs or new_g < g_costs[neighbor]:
                g_costs[neighbor] = new_g
                h = heuristic(neighbor, goal)
                new_f = new_g + h
                heapq.heappush(pq, (new_f, new_g, neighbor, path + [neighbor]))
                
    return None, closed_list, open_snapshots

# Execute search based on selection
if start_node == goal_node:
    st.warning("Start aur destination locations same nahi ho sakte!")
    path, closed_list, open_snapshots = [start_node], [start_node], [[start_node]]
else:
    if algorithm == "Breadth-First Search (BFS)":
        path, closed_list, open_snapshots = run_bfs(G_working, start_node, goal_node)
    elif algorithm == "Depth-First Search (DFS)":
        path, closed_list, open_snapshots = run_dfs(G_working, start_node, goal_node)
    else:
        path, closed_list, open_snapshots = run_astar(G_working, start_node, goal_node, weight_attr)

# Calculate total path cost
total_cost = 0
if path:
    for i in range(len(path) - 1):
        total_cost += G_working[path[i]][path[i+1]].get(weight_attr, 0)

# Main UI layout with Plotly and Metrics
col_map, col_info = st.columns([2, 1])

with col_map:
    st.subheader("🗺️ Campus Visual Map (Classrooms & Labs)")
    
    pos = nx.get_node_attributes(G, 'pos')
    
    edge_x, edge_y = [], []
    for edge in G_working.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=2, color='#cbd5e1'), hoverinfo='none', mode='lines')

    path_x, path_y = [], []
    if path and len(path) > 1:
        for i in range(len(path) - 1):
            x0, y0 = pos[path[i]]
            x1, y1 = pos[path[i+1]]
            path_x.extend([x0, x1, None])
            path_y.extend([y0, y1, None])
            
    path_trace = go.Scatter(x=path_x, y=path_y, line=dict(width=6, color='#4f46e5'), hoverinfo='none', mode='lines', name='Final Route')

    node_x, node_y, node_text, node_color = [], [], [], []
    for node, data in G.nodes(data=True):
        x, y = data['pos']
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}\n({data['type']})")
        
        if node == start_node:
            node_color.append('#10b981') # Green for Start
        elif node == goal_node:
            node_color.append('#ef4444') # Red for Goal
        elif node in path:
            node_color.append('#6366f1') # Blue for Path
        else:
            node_color.append('#94a3b8') # Gray for others

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=node_text, textposition="bottom center",
        marker=dict(size=18, color=node_color, line=dict(width=2, color='white'))
    )

    fig = go.Figure(data=[edge_trace, path_trace, node_trace],
                    layout=go.Layout(showlegend=False, hovermode='closest',
                                     margin=dict(b=0,l=0,r=0,t=0),
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    st.plotly_chart(fig, use_container_width=True)

with col_info:
    st.subheader("📊 Task 7: Agent States & Metrics")
    if path:
        st.success("Optimal Route Found Successfully!")
        st.metric(label=f"Total Path Cost ({optimization_metric})", value=total_cost)
        
        st.markdown("**Final Route Sequence (Actions):**")
        st.code(" ➔ ".join(path))
    else:
        st.error("Koi valid route nahi mila! Blocked paths check karein.")
        
    st.markdown("---")
    st.markdown("**Search Agent Logs:**")
    st.info(f"**Closed List (Expanded States):**\n{closed_list}")
    st.info(f"**Open List Snapshot:**\n{open_snapshots[-1] if open_snapshots else 'N/A'}")