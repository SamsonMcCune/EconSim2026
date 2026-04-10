from scipy.spatial import cKDTree
from Agent_Class import Agent
from Agent_Class import dictionary_to_vector
from Relationship_Class import Relationship
import matplotlib.pyplot as plt
import numpy as np
import time
from multiprocessing import Pool
np.random.seed(67)
target_growth = 0.03
pid_state = {
    "integral": 0.0,
    "prev_error": 0.0
}
prev_growth = 0
trait_keys = ["intelligence", "wisdom", "strength", "dexterity", "charisma", "comeliness", "constitution"]
param_keys = ["base_scale", "age_decay", "carrying_capacity", "fertility_rate", "avg_children"]
params = {
    "base_scale": 1.0,
    "age_decay": 0.05,
    "carrying_capacity": 5000,
    "fertility_rate": .7,
    "avg_children": 2.5,
    "Kp": 3.98098114906752 ,
    "Ki": 0.31778040091329735 ,
    "Kd": 2.195888086718796
}



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

def reproduce(marriages, trait_keys, next_id_start, params):
    children = []
    next_id = next_id_start
    for marriage in marriages:
        f = np.random.rand()
        if f > params["fertility_rate"]:
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
                orthogonal = np.array([-direction[1], direction[0]])
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
        values = [a.genome[trait] for a in agents]
        mean_val = np.mean(values)
        history["traits"][trait].append(mean_val)

def compute_pid_adjustment(error, pid_state, params):
    Kp = params["Kp"]
    Ki = params["Ki"]
    Kd = params["Kd"]
    pid_state["integral"] += error
    pid_state["integral"] = np.clip(pid_state["integral"], -5, 5)
    derivative = error - pid_state["prev_error"]
    pid_state["prev_error"] = error
    control = (Kp * error + Ki * pid_state["integral"] + Kd * derivative)
    control = np.clip(control, -2, 2)
    return control

def survives(agent, round, population_size, params, control):
    base_scale = params["base_scale"]
    age_decay = params["age_decay"]
    base_survival = base_scale * (agent.genome["constitution"] / 100)
    age = round - agent.generation
    age_effect = 1 / (1 + age_decay * age)
    K = params["carrying_capacity"]
    density_effect = 1 / (1 + (population_size / K)**2)
    adjusted_base = np.clip(base_survival * np.exp(2 * control), 0.01, 1.0)
    survival_prob = adjusted_base * age_effect * density_effect
    survival_prob = np.clip(survival_prob, 0.0, 1)
    return np.random.rand() < survival_prob

def population_evolution(num_agents, total_rounds, trait_keys, params):
    agents, _ = first_generation(num_agents)
    next_id = num_agents
    history = initialize_tracker(trait_keys)
    prev_pop = len(agents)
    prev_growth = 0.0
    pid_state = {"integral": 0.0, "prev_error": 0.0}
    relationships = {}
    control = 0.0 
    for t in range(total_rounds):
        print(f"Round {t}, population: {len(agents)}")
        record_population_state(agents, history, trait_keys)
        relationships = build_relationships_kdtree(agents)
        friendships = build_friendships(relationships)
        marriages = build_marriages(friendships)
        children, next_id = reproduce(marriages, trait_keys, next_id, params)
        agents.extend(children)
        agents = [a for a in agents if survives(a, t, len(agents), params, control)]
        current_pop = len(agents)
        raw_growth = (current_pop - prev_pop) / max(prev_pop, 1)
        growth_rate = 0.8 * prev_growth + 0.2 * raw_growth
        prev_growth = growth_rate
        error = target_growth - growth_rate
        control = compute_pid_adjustment(error, pid_state, params)
        print(f"   growth={growth_rate:.4f}, error={error:.4f}, control={control:.4f}")
        prev_pop = current_pop
        if len(agents) == 0:
            break
    return agents, history

def evaluate_pid(params, runs=3):
    errors = []
    for _ in range(runs):
        _, history = population_evolution(
            num_agents=1000,
            total_rounds=100,
            trait_keys=trait_keys,
            params=params
        )
        pop = np.array(history["population"])
        if len(pop) < 5:
            return 1e6
        growth = np.diff(pop) / np.maximum(pop[:-1], 1)
        target = 0.03
        mse = np.mean((growth - target) ** 2)
        volatility = np.std(growth)
        errors.append(mse + 0.5 * volatility)
    return np.mean(errors)

def random_pid():
    return {
        **params,  # keep your base params
        "Kp": np.random.uniform(1, 15),
        "Ki": np.random.uniform(0.0, 1.0),
        "Kd": np.random.uniform(0.0, 5.0)
    }

def mutate(p):
    new = p.copy()
    for k in ["Kp", "Ki", "Kd"]:
        new[k] = max(0, p[k] + np.random.normal(0, 0.2 * p[k]))
    return new


def crossover(p1, p2):
    return {
        k: (p1[k] + p2[k]) / 2 for k in p1
    }


def autotune_pid(generations=15, pop_size=10):
    population = [random_pid() for _ in range(pop_size)]
    for gen in range(generations):
        with Pool() as pool:
            scores = pool.map(evaluate_pid, population)
        scored = list(zip(scores, population))
        scored.sort(key=lambda x: x[0])
        best_score, best = scored[0]
        print(f"Gen {gen}: score={best_score:.4f}, Kp={best['Kp']:.2f}, Ki={best['Ki']:.2f}, Kd={best['Kd']:.2f}")
        survivors = [p for _, p in scored[:pop_size // 2]]
        new_population = survivors.copy()
        while len(new_population) < pop_size:
            p1, p2 = np.random.choice(survivors, 2, replace=False)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)
        population = new_population

    return best

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

def visualize_network(positions, relationships, subdivisions):
    plt.figure(figsize=(6,6))
    plt.scatter(positions[:, 0], positions[:, 1], zorder=2)
    if isinstance(relationships, dict):
        iterable = relationships.values()
    else:
        iterable = relationships
    for rel in iterable:
        if rel.friends:
            x_vals = [rel.a.position[0], rel.b.position[0]]
            y_vals = [rel.a.position[1], rel.b.position[1]]
            plt.plot(x_vals, y_vals, linewidth=0.5, alpha=0.5, zorder=1)
    cell_size = 1 / subdivisions
    for i in range(subdivisions + 1):
        plt.axhline(i * cell_size, linewidth=0.5)
        plt.axvline(i * cell_size, linewidth=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Agent Network")
    plt.show()

def visualize_children(agents, children, subdivisions=50):
    plt.figure(figsize=(6,6))
    agent_positions = np.array([a.position for a in agents])
    plt.scatter(agent_positions[:, 0], agent_positions[:, 1],
                color='lightgray', s=10, label='Agents', zorder=1)
    child_positions = np.array([c.position for c in children])
    plt.scatter(child_positions[:, 0], child_positions[:, 1],
                color='red', s=20, label='Children', zorder=2)
    cell_size = 1 / subdivisions
    for i in range(subdivisions + 1):
        plt.axhline(i * cell_size, linewidth=0.5, alpha=0.3)
        plt.axvline(i * cell_size, linewidth=0.5, alpha=0.3)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Children Locations")
    plt.legend()
    plt.show()
if __name__ == "__main__":
    best_params = autotune_pid()
    print("Best PID:", best_params["Kp"], best_params["Ki"], best_params["Kd"])
    start = time.time()
    num_agents = 1000
    agents, history = population_evolution(num_agents, 100, trait_keys, params)
    end = time.time()
    print("Time:", end - start, "seconds")
    plot_population(history)
    plot_traits(history)
