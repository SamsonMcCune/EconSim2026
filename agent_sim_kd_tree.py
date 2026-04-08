from scipy.spatial import cKDTree
from Agent_Class import Agent
from Agent_Class import dictionary_to_vector
from Relationship_Class import Relationship
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

def first_generation(num_agents):
    agents = [Agent(i, 1) for i in range(num_agents)]
    positions = np.array([agent.position for agent in agents])
    return agents, positions

def agent_learning(agents):
    for agent in agents:
        agent.learning()
    return agents

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

def reproduce(marriages, trait_keys, next_id_start, fertility_rate):
    children = []
    next_id = next_id_start
    for marriage in marriages:
        f = np.random.rand()
        if f > fertility_rate:
            continue
        else:
            parent_a, parent_b = marriage.a, marriage.b
            child_generation = parent_a.generation + 1
            num_children = np.random.poisson(2.1)
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
        values = [a.genome[trait] for a in agents]
        mean_val = np.mean(values)
        history["traits"][trait].append(mean_val)

def survives(agent, round, population_size):
    base_survival = agent.genome["constitution"] / 100
    age = round - agent.generation
    age_effect = np.exp(-0.05 * age)
    density = population_size / 2000000
    density_effect = np.exp(-0.7 * density)
    survival_prob = base_survival * age_effect * density_effect
    return np.random.rand() < survival_prob

def population_evolution(num_agents, total_rounds, trait_keys):
    agents, _ = first_generation(num_agents)
    next_id = num_agents
    history = initialize_tracker(trait_keys)
    for t in range(total_rounds):
        print(f"Round {t}, population: {len(agents)}")
        record_population_state(agents, history, trait_keys)
        relationships = build_relationships_kdtree(agents)
        friendships = build_friendships(relationships)
        marriages = build_marriages(friendships)
        children, next_id = reproduce(marriages, trait_keys, next_id, .4)
        agents.extend(children)
        agents = [a for a in agents if survives(a, t, len(agents))]
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
    trade_qty = max(1, int(trade_qty))
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
        return True
    return False

def bilateral_market(agents, periods, rounds, delta):
    N = len(agents)
    gini_rounds = []
    for l in range(periods):
        for _ in range(rounds):
            i, j = np.random.choice(N, 2, replace=False)
            agent_i_num = i
            agent_j_num = j
            t_cost = transaction_cost(agent_i_num, agent_j_num, num_agents)
            good = np.random.choice(["apple","barracuda"])
            trade(agents[i], agents[j], good, delta, price_history, t_cost)
        for agent in agents:
            agent.saving_appreciation(interest_rate)
            #print(f"Agent {agent.name} saved ${cash_saved:.2f}")
        apple_price = price_history["apple"][-1] if price_history["apple"] else None
        barracuda_price = price_history["barracuda"][-1] if price_history["barracuda"] else None
        wealth_per_period = get_total_wealth(agents,apple_price,barracuda_price)
        gini_rounds.append(gini(wealth_per_period))
    return agents, gini_rounds

def market_prices(price_history):
    market_price = {}
    for good, prices in price_history.items():
        if len(prices) > 0:
            market_price[good] = np.mean(prices)
        else:
            market_price[good] = None
    return market_price

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

start = time.time()
trait_keys = ["intelligence", "wisdom", "strength", "dexterity", "charisma", "comeliness", "constitution"]
agents, history = population_evolution(1000, 100, trait_keys)
end = time.time()
print("Time:", end - start, "seconds")
plot_population(history)
plot_traits(history)