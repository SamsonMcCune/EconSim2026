import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
import time
start = time.time()
np.random.seed(1)

class person:

    def __init__(self, name):
        self.id = name
        self.position = random_uniform_position()
        self.genome = {
            "gender": np.random.choice(["M","F"]),
            "intelligence": np.random.normal(100,15),
            "wisdom": 200*np.random.rand(),
            "strength": 200*np.random.rand(),
            "dexterity": 200*np.random.rand(),
            "charisma": 200*np.random.rand(),
            "comeliness": 200*np.random.rand(),
            "constitution": 200*np.random.rand()
        }
        self.radius = self.genome["dexterity"]/500

def preprocess_genomes(agents, keys):
    genome_matrix = np.array([
        dictionary_to_vector(agent.genome, keys)
        for agent in agents
    ]) / 100

    normalized = genome_matrix - 1
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    normalized = normalized / norms

    return normalized

def random_uniform_position():
    position = np.random.rand(2)
    return position

def random_normal_position(mean, variance):
    position = (np.random.normal(mean, variance, 2))
    return position

def first_generation(num_agents):
    agents = []
    for i in range(num_agents):
        agents.append(person(i))
    return agents

def dictionary_to_vector(traits, keys):
    return np.array([traits[k] for k in keys])

def cosine_similarity_fast(i, j, normalized_genomes):
    return np.dot(normalized_genomes[i], normalized_genomes[j])

def relationship_key(agent1, agent2):
    agent_1_2_key = tuple(sorted((agent1.id, agent2.id)))
    return agent_1_2_key

def relationship_rating_normalized(i, j, agents, normalized_genomes, rel_map):
    a = agents[i]
    b = agents[j]
    cosine_sim = cosine_similarity_fast(i, j, normalized_genomes)
    distance = get_distance(a, b)
    noise = np.random.normal(0, 0.15)
    distance_penalty = np.exp(-distance / (a.radius + b.radius))
    rel_score = np.clip(cosine_sim * distance_penalty + noise, -1, 1)
    rel_key = relationship_key(a, b)
    rel_map[rel_key] = {
        "distance": distance,
        "score": rel_score
    }
    return rel_map[rel_key]

def relationship_rating(agent1, agent2, rel_map):
    cosine_sim = cosine_similarity_fast(agent1, agent2, keys)
    distance = get_distance(agent1, agent2)
    noise = np.random.normal(0, 0.15)
    distance_penalty = np.exp(-distance / (agent1.radius + agent2.radius))
    rel_score = np.clip(cosine_sim * distance_penalty + noise, -1, 1)
    rel_key = relationship_key(agent1, agent2)
    rel_map[rel_key] = {
        "distance": distance,
        "score": rel_score
    }
    return rel_map[rel_key]

def get_distance(agent1, agent2):
    v1 = np.array(agent1.position)
    v2 = np.array(agent2.position)
    return np.linalg.norm(v1 - v2)

def marriage(G, threshold):
    edges = [
        (u, v, data["score"])
        for u, v, data in G.edges(data=True)
        if data["score"] > threshold
    ]
    edges.sort(key=lambda x: x[2], reverse=True)
    matched = set()
    marriages = []
    for u, v, score in edges:
        if agents[u].genome["gender"] == agents[v].genome["gender"]:
            continue
        if u not in matched and v not in matched:
            marriages.append((u, v))
            matched.add(u)
            matched.add(v)
    return marriages

def reproduce(G):

    children = []
    return children

def build_relationship_network_radius(agents, rel_map, radius=0.1):
    G = nx.Graph()
    for agent in agents:
        G.add_node(agent.id, obj=agent, pos=agent.position)
    positions = np.array([agent.position for agent in agents])
    nbrs = NearestNeighbors(radius=radius, algorithm='ball_tree').fit(positions)
    distances, indices = nbrs.radius_neighbors(positions)
    n = len(agents)
    for i in range(n):
        for j_idx in indices[i]:
            if i == j_idx:
                continue
            a = agents[i]
            b = agents[j_idx]
            rel_key = relationship_key(a, b)
            if rel_key not in rel_map:
                relationship_rating_normalized(i, j_idx, agents, normalized_genomes, rel_map)
            metrics = rel_map[rel_key]
            G.add_edge(
                a.id,
                b.id,
                score=metrics["score"],
                distance=metrics["distance"]
            )
    return G

def build_relationship_network_fast(agents, rel_map, radius=0.1):
    G = nx.Graph()

    positions = np.array([agent.position for agent in agents])
    tree = cKDTree(positions)

    # returns set of (i, j) pairs
    pairs = tree.query_pairs(r=radius)

    for i, j in pairs:
        a = agents[i]
        b = agents[j]

        rel_key = relationship_key(a, b)

        if rel_key not in rel_map:
            relationship_rating_normalized(i, j, agents, normalized_genomes, rel_map)

        metrics = rel_map[rel_key]

        G.add_edge(
            a.id,
            b.id,
            score=metrics["score"],
            distance=metrics["distance"]
        )
    return G

def visualize_network(agents, rel_map, marriages, threshold=0.5):
    G = nx.Graph()
    for agent in agents:
        G.add_node(agent.id, pos=agent.position)
    for (i, j), metrics in rel_map.items():
        if metrics["score"] > threshold:
            G.add_edge(i, j, weight=metrics["score"])
    pos = {agent.id: agent.position for agent in agents}
    marriage_set = {tuple(sorted(pair)) for pair in marriages}
    normal_edges = []
    marriage_edges = []
    for u, v in G.edges():
        if tuple(sorted((u, v))) in marriage_set:
            marriage_edges.append((u, v))
        else:
            normal_edges.append((u, v))
    nx.draw_networkx_nodes(G, pos, node_size=5)
    nx.draw_networkx_edges(
        G, pos,
        edgelist=normal_edges,
        alpha=0.2
    )
    nx.draw_networkx_edges(
        G, pos,
        edgelist=marriage_edges,
        edge_color="red",
        width=1.5
    )
    plt.show()

gamma = .1
marriage_threshold = .8
keys = ["intelligence", "wisdom", "strength", "dexterity", "charisma", "comeliness", "constitution"]
agents = first_generation(1000)
rel_map = {}
normalized_genomes = preprocess_genomes(agents, keys)
network = build_relationship_network_radius(agents, rel_map, radius=gamma)
scores = [metrics["score"] for metrics in rel_map.values()]
marriages = marriage(network, marriage_threshold)
visualize_network(agents, rel_map, marriages)
end = time.time()
print("Time:", end - start, "seconds")
print(f"There were {len(marriages)} marriages.")
plt.hist(scores, bins=50)
plt.xlabel("Relationship Score")
plt.ylabel("Frequency")
plt.title("Distribution of Relationship Scores")
plt.show()