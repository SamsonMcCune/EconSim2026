from scipy.spatial import cKDTree
from Agent_Class import Agent
from Agent_Class import dictionary_to_vector
from Relationship_Class import Relationship
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import time
np.random.seed(67)
trait_keys = ["intelligence", "wisdom", "strength", "dexterity", "charisma", "comeliness", "constitution"]
PARAM_KEYS = [
    "base_scale",
    "age_decay",
    "carrying_capacity",
    "fertility_rate",
    "avg_children"
]

def first_generation(num_agents):
    agents = [Agent(i, 1) for i in range(num_agents)]
    positions = np.array([agent.position for agent in agents])
    return agents, positions

def build_relationships_kdtree(agents):
    if len(agents) == 0:
        return {}
    positions = np.array([agent.position for agent in agents])
    tree = cKDTree(positions)
    max_radius = 0.1
    candidate_pairs = tree.query_pairs(r=max_radius)
    relationships = {}
    for i, j in candidate_pairs:
        a = agents[i]
        b = agents[j]
        dx = a.position[0] - b.position[0]
        dy = a.position[1] - b.position[1]
        dist2 = dx*dx + dy*dy
        r_sum = a.radius + b.radius
        if dist2 <= r_sum * r_sum:
            rel = Relationship(a, b)
            relationships[rel.key] = rel
    return relationships


def build_friendships(relationships):
    return [r for r in relationships.values() if r.friends]

def build_marriages(friendships):
    friendships.sort(key=lambda r: r.score, reverse=True)
    matched = set()
    marriages = []
    for rel in friendships:
        a, b = rel.a, rel.b
        if a.genome["gender"] == b.genome["gender"]:
            continue
        if a.id not in matched and b.id not in matched and a.generation == b.generation:
            rel.married = True
            marriages.append(rel)
            matched.add(a.id)
            matched.add(b.id)
    return marriages

def reproduce(marriages, trait_keys, next_id_start, fertility_rate, params):
    children = []
    next_id = next_id_start
    for marriage in marriages:
        f = np.random.rand()
        if f > fertility_rate:
            continue
        else:
            parent_a, parent_b = marriage.a, marriage.b
            child_generation = parent_a.generation + 1
            num_children = np.random.poisson(params["avg_children"])
            for n in range(num_children):
                child = Agent(next_id, child_generation)
                next_id += 1
                for trait in trait_keys:
                    p = np.random.rand()
                    child.genome[trait] = (
                        parent_a.genome[trait] * p +
                        parent_b.genome[trait] * (1 - p)
                    )
                direction = parent_b.position - parent_a.position
                orthogonal = np.array([-direction[1], direction[0]])  # perpendicular
                orthogonal /= np.linalg.norm(orthogonal) + 1e-8
                t = np.random.rand()
                base_pos = parent_a.position + t * direction
                offset = np.random.normal(0, 0.02)
                child.position = np.clip(base_pos + offset * orthogonal, 0, 1)
                child.radius = child.genome["dexterity"] / 200
                child.genome_vector = dictionary_to_vector(child.genome, trait_keys)
                child.genome_vector_normalized = (
                    child.genome_vector / np.linalg.norm(child.genome_vector)
                )
                children.append(child)
    return children, next_id

def initialize_tracker(trait_keys):
    history = {
        "population": [],
        "traits": {k: [] for k in trait_keys}
    }
    return history

def record_population_state(agents, history, trait_keys):
    history["population"].append(len(agents))
    for trait in trait_keys:
        if len(agents) == 0:
            history["traits"][trait].append(0)  # or np.nan
            continue
        values = [a.genome[trait] for a in agents]
        history["traits"][trait].append(np.mean(values))

def survives(agent, round, population_size, params):
    base_scale = params["base_scale"]
    age_decay = params["age_decay"]
    base_survival = base_scale * (agent.genome["constitution"] / 100)
    age = round - agent.generation
    age_effect = np.exp(-age_decay * age)
    K = params["carrying_capacity"]
    density_effect = max(0, 1 - population_size / K)
    survival_prob = base_survival * age_effect * density_effect
    return np.random.rand() < survival_prob

def target_population(t, P0=1000, growth_rate=0.03):
    return P0 * np.exp(growth_rate * t)

def evaluate_params(params, num_rounds=100, repeats=3):
    errors = []
    for _ in range(repeats):
        _, history = population_evolution_with_params(
            num_agents=300,
            total_rounds=num_rounds,
            trait_keys=trait_keys,
            params=params
        )
        actual = np.array(history["population"])
        target = np.array([target_population(t) for t in range(len(actual))])
        error = np.mean(((actual - target) / (target + 1)) ** 2)
        errors.append(error)
        actual = np.array(history["population"])
        if len(actual) == 0 or actual[-1] == 0:
            return 1e6
    return np.mean(errors)

def mutate(params):
    return {
        k: max(1e-5, v + np.random.normal(0, 0.1 * v))
        for k, v in params.items()
    }

def optimize():
    params = {
        "base_scale": 1.0,
        "age_decay": 0.05,
        "carrying_capacity": 5000,
        "fertility_rate": 0.4,
        "avg_children": 2.1
    }
    best_params = params
    best_score = evaluate_params(params, num_rounds=50, repeats=2)
    for gen in range(30):
        candidates = [mutate(best_params) for _ in range(10)] 
        scored = []
        for c in candidates:
            score = evaluate_params(c, num_rounds=50, repeats=2)
            scored.append((score, c))
        scored.sort(key=lambda x: x[0])
        best_score, best_params = scored[0]
        print(f"Gen {gen}: best_score={best_score:.6f}, params={best_params}")
    return best_params

def fast_optimize(model, n_iter=1000):
    best_params = None
    best_score = float("inf")

    for _ in range(n_iter):
        candidate = np.array([
            np.random.uniform(0.5, 1.5),
            np.random.uniform(0.01, 0.1),
            np.random.uniform(2000, 8000),
            np.random.uniform(0.2, 0.6),
            np.random.uniform(1.5, 3.0)
        ])
        pred = model.predict(candidate.reshape(1, -1))[0]
        if pred < best_score:
            best_score = pred
            best_params = candidate
    return best_params

def population_evolution_with_params(num_agents, total_rounds, trait_keys, params):
    agents, _ = first_generation(num_agents)
    next_id = num_agents
    history = initialize_tracker(trait_keys)
    for t in range(total_rounds):
        if len(agents) == 0:
            break
        record_population_state(agents, history, trait_keys)
        relationships = build_relationships_kdtree(agents)
        friendships = build_friendships(relationships)
        marriages = build_marriages(friendships)
        children, next_id = reproduce(marriages, trait_keys, next_id, params["fertility_rate"], params)
        agents.extend(children)
        agents = [a for a in agents if survives(a, t, len(agents), params)]
    return agents, history

def generate_training_data(n_samples=30):
    x = []
    y = []

    for i in range(n_samples):
        params = {
            "base_scale": np.random.uniform(0.5, 1.5),
            "age_decay": np.random.uniform(0.01, 0.1),
            "carrying_capacity": np.random.uniform(2000, 8000),
            "fertility_rate": np.random.uniform(0.2, 0.6),
            "avg_children": np.random.uniform(1.5, 3.0)
        }
        error = evaluate_params(params, num_rounds=20, repeats=1)
        x.append([params[k] for k in PARAM_KEYS])
        y.append(error)
        print(f"Sample {i + 1}: error={error:.4f}")
    return np.array(x), np.array(y)

def plot_population(history):
    plt.figure(figsize=(6,4))
    plt.plot(history["population"])
    plt.xlabel("Time step")
    plt.ylabel("Population size")
    plt.title("Population Over Time")
    plt.show()

def plot_traits(history):
    plt.figure(figsize=(8,5))

    for trait, values in history["traits"].items():
        plt.plot(values, label=trait)

    plt.xlabel("Time step")
    plt.ylabel("Average value")
    plt.title("Trait Evolution Over Time")
    plt.legend()
    plt.show()

def plot_vs_target(history):
    actual = np.array(history["population"])
    target = np.array([target_population(t) for t in range(len(actual))])

    plt.plot(actual, label="Actual")
    plt.plot(target, label="Target (Exponential)")
    plt.legend()
    plt.title("Population vs Target")
    plt.show()

start = time.time()
x, y = generate_training_data(100)
model = RandomForestRegressor(n_estimators=100)
model.fit(x, y)
best = fast_optimize(model)
param_dict = {
    "base_scale": best[0],
    "age_decay": best[1],
    "carrying_capacity": best[2],
    "fertility_rate": best[3],
    "avg_children": best[4]
}
real_score = evaluate_params(param_dict, num_rounds=100, repeats=3)
print("Final params:", param_dict)
print("Real score:", real_score)
end = time.time()
print("Time:", end - start, "seconds")