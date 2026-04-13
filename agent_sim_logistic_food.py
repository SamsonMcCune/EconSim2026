from scipy.spatial import cKDTree
from Agent_Class import Agent
from Agent_Class import dictionary_to_vector
from Relationship_Class import Relationship
import matplotlib.pyplot as plt
import numpy as np
import time
np.random.seed(67)

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

def initialize_tracker(trait_keys):
    return {
        "population": [],
        "food": [],
        "traits": {k: [] for k in trait_keys}
    }

def record_population_state(agents, history, trait_keys, food):
    history["population"].append(len(agents))
    history["food"].append(food)
    for trait in trait_keys:
        values = [a.genome[trait] for a in agents]
        history["traits"][trait].append(np.mean(values))

def survives(agent, round, necessary_food, consumed_food):
    base = agent.genome["constitution"] / 100
    age = round - agent.generation
    age_effect = np.exp(-0.03 * age)
    if necessary_food <= 0:
        food_effect = 1
    else:
        food_effect = np.clip(consumed_food / necessary_food, 0, 1)
    survival_prob = (0.5 * base + 0.3 * food_effect + 0.2 * age_effect)
    survival_prob = np.clip(survival_prob, 0, 1)
    return np.random.rand() < survival_prob

def reproduce(marriages, trait_keys, next_id_start):
    children = []
    next_id = next_id_start
    for marriage in marriages:
        parent_a, parent_b = marriage.a, marriage.b
        surplus = marriage.food_surplus
        eff_a = parent_a.genome["strength"] / (parent_a.genome["metabolic_rate"] + 1e-8)
        eff_b = parent_b.genome["strength"] / (parent_b.genome["metabolic_rate"] + 1e-8)
        efficiency = (eff_a + eff_b) / 2
        fertility_modifier = 0.2 + 0.4 * np.tanh(surplus) + 0.4 * np.tanh(efficiency / 10)
        if np.random.rand() > fertility_modifier:
            continue
        else:
            child_generation = parent_a.generation + 1
            expected_children = 2.1
            num_children = np.random.poisson(expected_children)
            for n in range(num_children):
                child = Agent(next_id, child_generation)
                next_id += 1
                for trait in trait_keys:
                    p = np.random.rand()
                    value = (
                        parent_a.genome[trait] * p +
                        parent_b.genome[trait] * (1 - p)
                    )
                    mutation_strength = 0.1
                    value += np.random.normal(0, mutation_strength * value)
                    child.genome[trait] = max(0, value)
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

def food_production(food, growth_rate, food_capacity):
    growth = growth_rate * food * (1 - food / food_capacity)
    return food + growth

def food_consumption(agents, food):
    strength_values = np.array([a.genome["strength"] for a in agents])
    weights = np.maximum(strength_values, 0)
    total_weight = np.sum(weights) + 1e-8
    shares = weights / total_weight * food
    consumed = {}
    required = {}
    food_balance = {}
    for agent, share in zip(agents, shares):
        need = (agent.genome["metabolic_rate"] / 100) ** 1.5
        actual = min(share, need)
        consumed[agent.id] = actual
        required[agent.id] = need
        food_balance[agent.id] = max(0, share - need)
    food_remaining = food - sum(consumed.values())
    return food_remaining, food_balance, consumed, required

def assign_food_to_relationships(relationships, food_balance):
    for rel in relationships.values():
        a_id = rel.a.id
        b_id = rel.b.id
        surplus_a = food_balance.get(a_id, 0)
        surplus_b = food_balance.get(b_id, 0)
        rel.food_surplus = 0.5 * rel.food_surplus + max(0, surplus_a + surplus_b)

def population_evolution_logistic(num_agents, initial_food, food_growth, total_rounds, trait_keys):
    agents, _ = first_generation(num_agents)
    next_id = num_agents
    history = initialize_tracker(trait_keys)
    food = initial_food
    food_capacity = initial_food * 5
    for round in range(total_rounds):
        relationships = build_relationships_kdtree(agents)
        food, food_balance, consumed, required = food_consumption(agents, food)
        assign_food_to_relationships(relationships, {aid: consumed[aid] - required[aid] for aid in consumed})
        print(f"Round {round + 1}, population: {len(agents)}")
        record_population_state(agents, history, trait_keys, food)
        friendships = build_friendships(relationships)
        marriages = build_marriages(friendships)
        children, next_id = reproduce(marriages, trait_keys, next_id)
        agents.extend(children)
        for c in children:
            consumed[c.id] = 0
            required[c.id] = min(c.genome["metabolic_rate"] / 100, 1)
        agents = [
            a for a in agents
            if survives(a, round, required[a.id], consumed[a.id])
        ]
        food = food_production(food, food_growth, food_capacity)
    return agents, history, food

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

def plot_population_and_food(history):
    fig, ax1 = plt.subplots(figsize=(7,5))

    ax1.plot(history["population"], label="Population")
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Population")

    ax2 = ax1.twinx()
    ax2.plot(history["food"], linestyle='--', label="Food")
    ax2.set_ylabel("Food")

    fig.suptitle("Population and Food Over Time")

    # combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2)

    plt.show()

start = time.time()
trait_keys = ["intelligence", "wisdom", "strength", "dexterity", "charisma", "comeliness", "constitution", "metabolic_rate"]
num_agents = 100
initial_food = 10*num_agents
food_growth = num_agents/initial_food
agents, history, food = population_evolution_logistic(num_agents, initial_food, food_growth, 35, trait_keys)
end = time.time()
print("Time:", end - start, "seconds")
plot_population(history)
plot_traits(history)