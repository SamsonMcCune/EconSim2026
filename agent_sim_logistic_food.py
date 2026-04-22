from scipy.spatial import cKDTree
from Agent_Class import Agent
from Agent_Class import dictionary_to_vector
from Relationship_Class import Relationship
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
import time
np.random.seed(67)

base_prices = {
    "apple": 2,
    "barracuda": 5
}

price_history = {
    "apple": [],
    "barracuda": []
}

def normalize(grid):
    return 0.5 + (grid - grid.min()) / (grid.max() - grid.min())

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
    marriages = []
    for rel in friendships:
        a, b = rel.a, rel.b
        if a.genome["gender"] == b.genome["gender"]:
            continue
        if (
            not a.married and
            not b.married and
            a.generation == b.generation
        ):
            rel.married = True
            a.married = True
            b.married = True
            a.partner_id = b.id
            b.partner_id = a.id
            marriages.append(rel)
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


def survives(agent, round, consumed_food):
    agent.energy += consumed_food
    metabolic_cost = (agent.genome["metabolic_rate"] / 100) ** 1.5
    agent.energy -= metabolic_cost
    age = round - agent.generation
    age_cost = 0.01 * age
    agent.energy -= age_cost
    constitution = agent.genome["constitution"] / 100
    agent.energy += 0.2 * constitution * metabolic_cost  # partial refund-
    if agent.energy <= 0:
        return False
    return True

def reproduce(marriages, trait_keys, next_id_start):
    children = []
    next_id = next_id_start
    for marriage in marriages:
        parent_a, parent_b = marriage.a, marriage.b
        surplus = marriage.food_surplus
        reproduction_cost_a = 0.5 * (parent_a.genome["metabolic_rate"] / 100)
        reproduction_cost_b = 0.5 * (parent_b.genome["metabolic_rate"] / 100)
        if parent_a.energy > reproduction_cost_a and parent_b.energy > reproduction_cost_b:
            parent_a.energy -= reproduction_cost_a / 2
            parent_b.energy -= reproduction_cost_b / 2
            avg_fitness = (parent_a.fitness + parent_b.fitness) / 2
            fertility_modifier = (0.2 * np.tanh(surplus) + 0.7 * np.tanh(avg_fitness / 100))
            if np.random.rand() > fertility_modifier:
                continue
            else:
                child_generation = parent_a.generation + 1
                surplus_a = max(0, parent_a.energy - 1.0)
                surplus_b = max(0, parent_b.energy - 1.0)
                available_energy = surplus_a + surplus_b
                num_children = int(available_energy / ((reproduction_cost_a + reproduction_cost_b)/2))
                energy_per_child = available_energy / (num_children + 1e-8)
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
                    child.energy = 0.5 * energy_per_child
                    child.inventory = np.array([0.0, 0.0, 1.0])
                    child.married = False
                    child.partner_id = None
                    children.append(child)
    return children, next_id

def random_mating(agents, trait_keys, next_id, rate=0.01, marriage_penalty=0.3):
    children = []
    n = len(agents)
    num_attempts = int(rate * n)
    for _ in range(num_attempts):
        parent_a, parent_b = np.random.choice(agents, 2, replace=False)
        if parent_a.genome["gender"] == parent_b.genome["gender"]:
            continue
        prob_a = marriage_penalty if getattr(parent_a, "married", False) else 1.0
        prob_b = marriage_penalty if getattr(parent_b, "married", False) else 1.0
        if np.random.rand() > prob_a or np.random.rand() > prob_b:
            continue
        rel = Relationship(parent_a, parent_b)
        if rel.distance > 0.1:
            continue
        fitness_a = parent_a.fitness
        fitness_b = parent_b.fitness
        fitness_prob = np.tanh((fitness_a + fitness_b) / 200)
        if np.random.rand() > fitness_prob:
            continue
        reproduction_cost_a = 0.5 * (parent_a.genome["metabolic_rate"] / 100)
        reproduction_cost_b = 0.5 * (parent_b.genome["metabolic_rate"] / 100)
        if parent_a.energy <= reproduction_cost_a or parent_b.energy <= reproduction_cost_b:
            continue
        surplus_a = max(0, parent_a.energy - 1.0)
        surplus_b = max(0, parent_b.energy - 1.0)
        available_energy = surplus_a + surplus_b
        avg_cost = (reproduction_cost_a + reproduction_cost_b) / 2
        num_children = int(available_energy / (avg_cost + 1e-8))
        if num_children <= 0:
            continue
        parent_a.energy -= reproduction_cost_a / 2
        parent_b.energy -= reproduction_cost_b / 2
        energy_per_child = available_energy / (num_children + 1e-8)
        for _ in range(num_children):
            child = Agent(next_id, max(parent_a.generation, parent_b.generation) + 1)
            next_id += 1
            for trait in trait_keys:
                p = np.random.rand()
                value = (
                    parent_a.genome[trait] * p +
                    parent_b.genome[trait] * (1 - p)
                )
                value += np.random.normal(0, 0.1 * value)
                child.genome[trait] = max(0, value)
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
            child.inventory = np.array([0.0, 0.0, 1.0])
            child.energy = 0.5 * energy_per_child
            child.married = False
            child.partner_id = None
            children.append(child)
    return children, next_id

def select_trade_pairs(relationships):
    rels = list(relationships.values())
    np.random.shuffle(rels)
    rels.sort(key=lambda r: getattr(r, "score", 0), reverse=True)
    used = set()
    trade_pairs = []
    for rel in rels:
        a_id = rel.a.id
        b_id = rel.b.id
        if a_id in used or b_id in used:
            continue
        trade_pairs.append(rel)
        used.add(a_id)
        used.add(b_id)
    return trade_pairs

def food_production(food, growth_rate, food_capacity):
    growth = growth_rate * food * (1 - food / food_capacity)
    return food + growth

def produce(agent, resource_grid):
    x = agent.x
    y = agent.y
    apple_yield = resource_grid["apple"][x, y]
    fish_yield  = resource_grid["barracuda"][x, y]
    strength = agent.genome["strength"] / 100
    metabolism = agent.genome["metabolic_rate"] / 100
    apples = strength * apple_yield / (metabolism + 1e-8)
    fish   = strength * fish_yield / (metabolism + 1e-8)
    agent.inventory[0] += apples
    agent.inventory[1] += fish

def food_consumption(agents, food):
    strength_values = np.array([a.genome["strength"] for a in agents])
    metabolic_values = np.array([a.genome["metabolic_rate"] for a in agents])
    weights = (strength_values/metabolic_values) ** 1.5
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

def consume_from_inventory(agent):
    apple_energy = 1.0
    fish_energy = 1.2
    need = (agent.genome["metabolic_rate"] / 100) ** 1.5
    apples_available = max(0.0, agent.inventory[0])
    fish_available = max(0.0, agent.inventory[1])
    apples_eaten = min(apples_available, need / apple_energy)
    energy_gained = apples_eaten * apple_energy
    remaining_need = max(0.0, need - energy_gained)
    fish_eaten = min(fish_available, remaining_need / fish_energy)
    energy_gained += fish_eaten * fish_energy
    agent.inventory[0] -= apples_eaten
    agent.inventory[1] -= fish_eaten

    surplus = max(0.0, agent.inventory[0] + agent.inventory[1])
    return energy_gained, surplus

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
    food_capacity = 5*len(agents)
    marriages = []
    for round in range(total_rounds):
        food_capacity = 0.2 * food_capacity + 0.8 * (6 * len(agents))
        relationships = build_relationships_kdtree(agents)
        market_pairs = select_trade_pairs(relationships)
        food, food_balance, consumed, required = food_consumption(agents, food)
        assign_food_to_relationships(relationships, {aid: consumed[aid] - required[aid] for aid in consumed})
        print(f"Round {round + 1}, population: {len(agents)}")
        record_population_state(agents, history, trait_keys, food)
        friendships = build_friendships(relationships)
        new_marriages = build_marriages(friendships)
        marriages += new_marriages
        alive_ids = {a.id for a in agents}
        marriages = [
            m for m in marriages
            if m.a.id in alive_ids and m.b.id in alive_ids
        ]
        children1, next_id = reproduce(marriages, trait_keys, next_id)
        children2, next_id = random_mating(agents, trait_keys, next_id, rate=0.01)
        children = children1 + children2
        agents.extend(children)
        for c in children:
            consumed[c.id] = 0
        agents = [
            a for a in agents
            if survives(a, round, consumed[a.id])
        ]
        food = food + food_production(food, food_growth, food_capacity)
    return agents, history, food

def population_evolution_market(num_agents, total_rounds, trait_keys, resource_grid, delta):
    agents, _ = first_generation(num_agents)
    next_id = num_agents
    history = initialize_tracker(trait_keys)
    marriages = []

    for round in range(total_rounds):
        print(f"Round {round + 1}, population: {len(agents)}")
        for a in agents:
            produce(a, resource_grid)
        relationships = build_relationships_kdtree(agents)
        market_pairs = select_trade_pairs(relationships)
        trades_completed = bilateral_market(market_pairs, delta, price_history)
        consumed = {}
        surplus_by_agent = {}
        required = {}
        for a in agents:
            energy_gained, surplus = consume_from_inventory(a)
            consumed[a.id] = energy_gained
            surplus_by_agent[a.id] = surplus
            required[a.id] = (a.genome["metabolic_rate"] / 100) ** 1.5
        assign_food_to_relationships(relationships, surplus_by_agent)
        before_survival_count = len(agents)
        agents = [a for a in agents if survives(a, round, consumed[a.id])]
        alive_ids = {a.id for a in agents}
        after_survival_count = len(alive_ids)
        marriages = [m for m in marriages if m.a.id in alive_ids and m.b.id in alive_ids]
        relationships = build_relationships_kdtree(agents)
        friendships = build_friendships(relationships)
        new_marriages = build_marriages(friendships)
        marriages += new_marriages
        children1, next_id = reproduce(marriages, trait_keys, next_id)
        children2, next_id = random_mating(agents, trait_keys, next_id, rate=0.01)
        children = children1 + children2
        agents.extend(children)
        total_food_stock = total_food_inventory(agents)
        record_population_state(agents, history, trait_keys, total_food_stock)
        num_agents = len(agents)
        avg_energy = np.mean([a.energy for a in agents]) if agents else 0
        avg_food = np.mean([a.inventory[0] + a.inventory[1] for a in agents]) if agents else 0
        births = len(children)
        deaths = before_survival_count - after_survival_count
        print(f"Trades: {trades_completed}, Births: {births}, Deaths: {deaths}, Avg Energy: {avg_energy:.2f}, Avg Food: {avg_food:.2f}")
    return agents, history

def determine_price(good, mrs_1, mrs_2, price_history):
    reservation_price = (mrs_1 + mrs_2) / 2
    n = len(price_history[good])
    if n == 0:
        price = reservation_price
    else:
        last_price = price_history[good][-1]
        price = last_price + (1/(n+1)) * (reservation_price - last_price)
    return price

def trade(agent_1, agent_2, good, delta, price_history, t_cost):
    goods_map = {"apple": 0, "barracuda": 1}
    if good not in goods_map:
        return "This is not a valid good."
    i = goods_map[good]
    mrs_1 = agent_1.mrs()[i]
    mrs_2 = agent_2.mrs()[i]
    if abs(mrs_1 - mrs_2) < 1e-8:
        return False
    if mrs_1 > mrs_2:
        buyer, seller = agent_1, agent_2
    else:
        buyer, seller = agent_2, agent_1
    price = determine_price(good, mrs_1, mrs_2, price_history)
    buyer_price = price + t_cost
    seller_price = price - t_cost
    mrs_gap = abs(mrs_1 - mrs_2)
    trade_qty = min(delta * mrs_gap, seller.inventory[i] * 0.25)
    if trade_qty <= 1e-6:
        return False
    cash_trade = buyer_price * trade_qty
    seller_revenue = seller_price * trade_qty
    if seller.inventory[i] < trade_qty:
        return False
    if buyer.inventory[2] < cash_trade:
        return False
    proposed_buyer_inventory = buyer.inventory.copy()
    proposed_seller_inventory = seller.inventory.copy()
    proposed_buyer_inventory[i] += trade_qty
    proposed_buyer_inventory[2] -= cash_trade
    proposed_seller_inventory[i] -= trade_qty
    proposed_seller_inventory[2] += seller_revenue
    if np.any(proposed_buyer_inventory <= 0) or np.any(proposed_seller_inventory <= 0):
        return False
    u_buyer_old = buyer.log_true_utility(buyer.inventory)
    u_seller_old = seller.log_true_utility(seller.inventory)
    u_buyer_new = buyer.log_expected_utility(proposed_buyer_inventory)
    u_seller_new = seller.log_expected_utility(proposed_seller_inventory)
    if u_buyer_new > u_buyer_old and u_seller_new > u_seller_old:
        buyer.inventory = proposed_buyer_inventory
        seller.inventory = proposed_seller_inventory
        price_history[good].append(price)
        print(f"Agent {buyer.id} bought {trade_qty} {good} from agent {seller.id} for ${price:.2f}")
        return True
    return False

def bilateral_market(market_pairs, delta, price_history):
    trades_completed = 0
    for rel in market_pairs:
        a = rel.a
        b = rel.b
        t_cost = rel.distance
        good = np.random.choice(["apple", "barracuda"])
        success = trade(a, b, good, delta, price_history, t_cost)
        if success:
            trades_completed += 1
    return trades_completed

def total_food_inventory(agents):
    return sum(a.inventory[0] + a.inventory[1] for a in agents)

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

grid_size = 50
raw_apple = np.random.rand(grid_size, grid_size)
resource_grid = {
    "apple": normalize(gaussian_filter(raw_apple, sigma=5)),
    "barracuda": normalize(gaussian_filter(1 - raw_apple, sigma=5))
}

start = time.time()
trait_keys = ["intelligence", "wisdom", "strength", "dexterity", "charisma", "comeliness", "constitution", "metabolic_rate"]
num_agents = 1000
initial_food = 10*num_agents
food_growth = num_agents/initial_food
agents, history = population_evolution_market(num_agents, 35, trait_keys, resource_grid, 0.1)
end = time.time()
print("Time:", end - start, "seconds")
plot_population(history)
plot_traits(history)