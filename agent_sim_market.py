from scipy.spatial import cKDTree
from Agent_Class import Agent
from Agent_Class import dictionary_to_vector
from Relationship_Class import Relationship
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
import time
np.random.seed(67)

# Functions
def agent_learning(agents):
    for agent in agents:
        agent.learning()
    return agents

def assign_food_to_relationships(relationships, food_balance):
    for rel in relationships.values():
        a_id = rel.a.id
        b_id = rel.b.id
        surplus_a = food_balance.get(a_id, 0)
        surplus_b = food_balance.get(b_id, 0)
        rel.food_surplus = 0.5 * rel.food_surplus + max(0, surplus_a + surplus_b)

def bilateral_market(market_pairs, delta, price_history):
    trades_completed = 0
    trade_log = []
    for rel in market_pairs:
        a = rel.a
        b = rel.b
        t_cost = rel.distance
        good = choose_trade_good(a, b)
        result = trade(a, b, good, delta, price_history, t_cost)
        trade_log.append(result)
        if result is not None and result.get("success", False):
            trades_completed += 1
    return trades_completed, trade_log

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
            abs(a.birth_round - b.birth_round) <= 2
        ):
            rel.married = True
            a.married = True
            b.married = True
            a.partner_id = b.id
            b.partner_id = a.id
            marriages.append(rel)
    return marriages

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

def build_empirical_supply_demand(order_history, good, round_idx=None, bins=40):
    if round_idx is None:
        records = order_history[good]
    else:
        records = [r for r in order_history[good] if r.get("round") == round_idx]

    buyer_prices = []
    buyer_qtys = []
    seller_prices = []
    seller_qtys = []

    for r in records:
        qty = r.get("quantity", 0.0)
        if qty <= 1e-8:
            continue

        buyer_prices.append(r["buyer_price"])
        buyer_qtys.append(qty)

        seller_prices.append(r["seller_price"])
        seller_qtys.append(qty)

    if len(buyer_prices) == 0 and len(seller_prices) == 0:
        return None, None, None

    all_prices = buyer_prices + seller_prices
    pmin = max(0.001, min(all_prices))
    pmax = max(pmin + 1e-3, max(all_prices))
    price_grid = np.linspace(pmin, pmax, bins)

    # Demand at p = total attempted buy quantity willing to pay at least p
    demand = np.array([
        sum(q for bp, q in zip(buyer_prices, buyer_qtys) if bp >= p)
        for p in price_grid
    ])

    # Supply at p = total attempted sell quantity willing to accept at most p
    supply = np.array([
        sum(q for sp, q in zip(seller_prices, seller_qtys) if sp <= p)
        for p in price_grid
    ])

    return price_grid, demand, supply

def build_empirical_supply_demand_filtered(order_history, good, round_idx=None, bins=40, include_reasons=None):
    if round_idx is None:
        records = order_history[good]
    else:
        records = [r for r in order_history[good] if r.get("round") == round_idx]

    if include_reasons is not None:
        records = [r for r in records if r["reason"] in include_reasons]

    buyer_prices = []
    buyer_qtys = []
    seller_prices = []
    seller_qtys = []

    for r in records:
        qty = r.get("quantity", 0.0)
        if qty <= 1e-8:
            continue
        buyer_prices.append(r["buyer_price"])
        buyer_qtys.append(qty)
        seller_prices.append(r["seller_price"])
        seller_qtys.append(qty)

    if len(buyer_prices) == 0 and len(seller_prices) == 0:
        return None, None, None

    all_prices = buyer_prices + seller_prices
    pmin = max(0.001, min(all_prices))
    pmax = max(pmin + 1e-3, max(all_prices))
    price_grid = np.linspace(pmin, pmax, bins)

    demand = np.array([
        sum(q for bp, q in zip(buyer_prices, buyer_qtys) if bp >= p)
        for p in price_grid
    ])
    supply = np.array([
        sum(q for sp, q in zip(seller_prices, seller_qtys) if sp <= p)
        for p in price_grid
    ])

    return price_grid, demand, supply

def choose_trade_good(a, b):
    gap_apple = abs(a.mrs()[0] - b.mrs()[0])
    gap_barracuda = abs(a.mrs()[1] - b.mrs()[1])
    return "apple" if gap_apple >= gap_barracuda else "barracuda"

def compute_supply_demand_curves(agents, good, price_grid):
    goods_map = {"apple": 0, "barracuda": 1}
    i = goods_map[good]
    demand = []
    supply = []

    for p in price_grid:
        total_demand = 0.0
        total_supply = 0.0

        for agent in agents:
            mrs = agent.mrs()[i]
            inventory_good = max(0.0, agent.inventory[i])
            cash = max(0.0, agent.inventory[2])

            if mrs > p:
                desired_qty = 0.25 * (mrs - p)
                affordable_qty = cash / (p + 1e-8)
                qty = min(desired_qty, affordable_qty)
                total_demand += max(0.0, qty)

            elif mrs < p:
                desired_qty = 0.25 * (p - mrs)
                qty = min(desired_qty, inventory_good)
                total_supply += max(0.0, qty)

        demand.append(total_demand)
        supply.append(total_supply)

    return np.array(demand), np.array(supply)

def consume_from_inventory(agent):
    apple_energy = 1.0
    fish_energy = 2.5
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

def determine_price(good, mrs_1, mrs_2, price_history):
    reservation_price = (mrs_1 + mrs_2) / 2
    n = len(price_history[good])
    if n == 0:
        price = reservation_price
    else:
        last_price = price_history[good][-1]
        price = last_price + (1/(n+1)) * (reservation_price - last_price)
    return price

def first_generation(num_agents):
    agents = [Agent(i, 1) for i in range(num_agents)]
    for agent in agents:
        agent.birth_round = 0
    positions = np.array([agent.position for agent in agents])
    return agents, positions

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

def food_production(food, growth_rate, food_capacity):
    growth = growth_rate * food * (1 - food / food_capacity)
    return food + growth

def initialize_tracker(trait_keys):
    return {
        "population": [],
        "food": [],
        "traits": {k: [] for k in trait_keys}
    }

def normalize(grid):
    return 0.5 + (grid - grid.min()) / (grid.max() - grid.min())

def population_evolution_market(num_agents, total_rounds, trait_keys, resource_grid, delta, market_history, order_history, snapshot_history):
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
        record_market_state(agents, market_history)
        trades_completed, trade_log = bilateral_market(market_pairs, delta, price_history)
        record_order_flow(trade_log, order_history, round)
        agents = agent_learning(agents)
        record_trade_log(trade_log, market_history)
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
        children1, next_id = reproduce(marriages, trait_keys, next_id, round)
        children2, next_id = random_mating(agents, trait_keys, next_id, round, rate=0.01)
        children = children1 + children2
        agents.extend(children)
        total_food_stock = total_food_inventory(agents)
        record_population_state(agents, history, trait_keys, total_food_stock)
        record_population_snapshot(agents, snapshot_history)
        num_agents = len(agents)
        avg_energy = np.mean([a.energy for a in agents]) if agents else 0
        avg_food = np.mean([a.inventory[0] + a.inventory[1] for a in agents]) if agents else 0
        births = len(children)
        deaths = before_survival_count - after_survival_count
        avg_age = np.mean([round - a.birth_round for a in agents]) if agents else 0
        print(f"Trades: {trades_completed}, Births: {births}, Deaths: {deaths}, Avg Energy: {avg_energy:.2f}, Avg Food: {avg_food:.2f}, Avg Age: {avg_age:.2f}")
    return agents, history, market_history, order_history, snapshot_history

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

def random_mating(agents, trait_keys, next_id, current_round, rate=0.01, marriage_penalty=0.3):
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
            child = Agent(next_id, current_round)
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

def record_population_state(agents, history, trait_keys, food):
    history["population"].append(len(agents))
    history["food"].append(food)
    for trait in trait_keys:
        values = [a.genome[trait] for a in agents]
        history["traits"][trait].append(np.mean(values))

def record_market_state(agents, market_history, goods=("apple", "barracuda"), num_prices=50):
    for good in goods:
        mrs_vals = []
        goods_map = {"apple": 0, "barracuda": 1}
        i = goods_map[good]

        for agent in agents:
            mrs = agent.mrs()[i]
            if np.isfinite(mrs):
                mrs_vals.append(mrs)

        if len(mrs_vals) == 0:
            price_grid = np.linspace(0.01, 10.0, num_prices)
        else:
            low = max(0.01, np.percentile(mrs_vals, 5))
            high = max(low + 1e-3, np.percentile(mrs_vals, 95))
            price_grid = np.linspace(low, high, num_prices)

        demand, supply = compute_supply_demand_curves(agents, good, price_grid)

        market_history[good]["prices"].append(price_grid)
        market_history[good]["demand_curves"].append(demand)
        market_history[good]["supply_curves"].append(supply)

def record_order_flow(trade_log, order_history, current_round):
    for trade in trade_log:
        trade = trade.copy()
        trade["round"] = current_round
        order_history[trade["good"]].append(trade)

def record_population_snapshot(agents, snapshot_history):
    snapshot_history.append({
        "positions": np.array([a.position.copy() for a in agents]),
        "birth_rounds": np.array([getattr(a, "birth_round", 0) for a in agents]),
        "energies": np.array([a.energy for a in agents]),
        "ids": np.array([a.id for a in agents])
    })

def record_trade_log(trade_log, market_history):
    goods = ["apple", "barracuda"]

    for good in goods:
        good_trades = [t for t in trade_log if t["good"] == good and t["success"]]

        if len(good_trades) == 0:
            market_history[good]["executed_prices"].append(np.nan)
            market_history[good]["executed_qty"].append(0.0)
        else:
            qtys = np.array([t["quantity"] for t in good_trades])
            prices = np.array([t["mid_price"] for t in good_trades])

            weighted_price = np.sum(prices * qtys) / (np.sum(qtys) + 1e-8)
            total_qty = np.sum(qtys)

            market_history[good]["executed_prices"].append(weighted_price)
            market_history[good]["executed_qty"].append(total_qty)

def reproduce(marriages, trait_keys, next_id_start, current_round):
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
                surplus_a = max(0, parent_a.energy - 1.0)
                surplus_b = max(0, parent_b.energy - 1.0)
                available_energy = surplus_a + surplus_b
                num_children = int(available_energy / ((reproduction_cost_a + reproduction_cost_b)/2))
                energy_per_child = available_energy / (num_children + 1e-8)
                for n in range(num_children):
                    child = Agent(next_id, current_round)
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

def survives(agent, current_round, consumed_food):
    agent.energy += consumed_food
    metabolic_cost = (agent.genome["metabolic_rate"] / 100) ** 1.5
    agent.energy -= metabolic_cost
    age = max(0, current_round - agent.birth_round)
    age_cost = 0.01 * age
    agent.energy -= age_cost
    constitution = agent.genome["constitution"] / 100
    agent.energy += 0.2 * constitution * metabolic_cost
    return agent.energy > 0

def total_food_inventory(agents):
    return sum(a.inventory[0] + a.inventory[1] for a in agents)

def trade(agent_1, agent_2, good, delta, price_history, t_cost):
    goods_map = {"apple": 0, "barracuda": 1}
    if good not in goods_map:
        return {
            "good": good,
            "success": False,
            "reason": "invalid_good"
        }
    i = goods_map[good]
    mrs_1 = agent_1.mrs()[i]
    mrs_2 = agent_2.mrs()[i]
    if abs(mrs_1 - mrs_2) < 1e-8:
        return {
            "good": good,
            "success": False,
            "reason": "no_mrs_gap"
        }
    if mrs_1 > mrs_2:
        buyer, seller = agent_1, agent_2
    else:
        buyer, seller = agent_2, agent_1
    price = determine_price(good, mrs_1, mrs_2, price_history)
    buyer_price = price + t_cost
    seller_price = price - t_cost
    mrs_gap = abs(mrs_1 - mrs_2)
    proposed_qty = min(delta * mrs_gap, seller.inventory[i] * 0.25)
    trade_record = {
        "good": good,
        "buyer_id": buyer.id,
        "seller_id": seller.id,
        "mid_price": price,
        "buyer_price": buyer_price,
        "seller_price": seller_price,
        "quantity": proposed_qty,
        "t_cost": t_cost,
        "buyer_cash": buyer.inventory[2],
        "seller_inventory": seller.inventory[i],
        "buyer_mrs": buyer.mrs()[i],
        "seller_mrs": seller.mrs()[i],
        "success": False,
        "reason": None
    }
    if proposed_qty <= 1e-6:
        trade_record["reason"] = "qty_too_small"
        return trade_record
    cash_trade = buyer_price * proposed_qty
    seller_revenue = seller_price * proposed_qty
    if seller.inventory[i] < proposed_qty:
        trade_record["reason"] = "seller_lacks_inventory"
        return trade_record
    if buyer.inventory[2] < cash_trade:
        trade_record["reason"] = "buyer_lacks_cash"
        return trade_record
    proposed_buyer_inventory = buyer.inventory.copy()
    proposed_seller_inventory = seller.inventory.copy()
    proposed_buyer_inventory[i] += proposed_qty
    proposed_buyer_inventory[2] -= cash_trade
    proposed_seller_inventory[i] -= proposed_qty
    proposed_seller_inventory[2] += seller_revenue
    if np.any(proposed_buyer_inventory <= 0) or np.any(proposed_seller_inventory <= 0):
        trade_record["reason"] = "nonpositive_inventory"
        return trade_record
    u_buyer_old = buyer.log_true_utility(buyer.inventory)
    u_seller_old = seller.log_true_utility(seller.inventory)
    u_buyer_new = buyer.log_expected_utility(proposed_buyer_inventory)
    u_seller_new = seller.log_expected_utility(proposed_seller_inventory)
    if not (u_buyer_new > u_buyer_old):
        trade_record["reason"] = "buyer_utility_fail"
        return trade_record
    if not (u_seller_new > u_seller_old):
        trade_record["reason"] = "seller_utility_fail"
        return trade_record
    buyer.inventory = proposed_buyer_inventory
    seller.inventory = proposed_seller_inventory
    price_history[good].append(price)
    trade_record["success"] = True
    trade_record["reason"] = "executed"
    print(f"Agent {buyer.id} bought {proposed_qty:.3f} {good} from agent {seller.id} for ${price:.2f}")
    return trade_record

# Graph Functions
def plot_market_evolution(market_history, good):
    fig, ax1 = plt.subplots(figsize=(8,5))

    prices = market_history[good]["executed_prices"]
    qtys = market_history[good]["executed_qty"]

    ax1.plot(prices, label="Executed Price")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Price")

    ax2 = ax1.twinx()
    ax2.plot(qtys, linestyle="--", label="Executed Quantity")
    ax2.set_ylabel("Quantity")

    fig.suptitle(f"Market Evolution for {good}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)

    plt.show()

def plot_empirical_supply_demand_filtered(order_history, good, round_idx=None, bins=40, include_reasons=None):
    price_grid, demand, supply = build_empirical_supply_demand_filtered(
        order_history, good, round_idx, bins, include_reasons
    )

    if price_grid is None:
        print(f"No order data for {good}.")
        return

    plt.figure(figsize=(7,5))
    plt.plot(price_grid, demand, label="Empirical Demand")
    plt.plot(price_grid, supply, label="Empirical Supply")

    title = f"Filtered Empirical Supply and Demand for {good}"
    if round_idx is not None:
        title += f" at Round {round_idx + 1}"

    plt.xlabel("Price")
    plt.ylabel("Quantity")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_empirical_supply_demand(order_history, good, round_idx=None, bins=40):
    price_grid, demand, supply = build_empirical_supply_demand(order_history, good, round_idx, bins)

    if price_grid is None:
        print(f"No order data for {good}.")
        return

    plt.figure(figsize=(7,5))
    plt.plot(price_grid, demand, label="Empirical Demand")
    plt.plot(price_grid, supply, label="Empirical Supply")

    title = f"Empirical Supply and Demand for {good}"
    if round_idx is not None:
        title += f" at Round {round_idx + 1}"

    plt.xlabel("Price")
    plt.ylabel("Quantity")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_order_scatter(order_history, good, round_idx=None):
    if round_idx is None:
        records = order_history[good]
    else:
        records = [r for r in order_history[good] if r.get("round") == round_idx]

    if len(records) == 0:
        print(f"No order data for {good}.")
        return

    buy_x_success = []
    buy_y_success = []
    buy_x_fail = []
    buy_y_fail = []

    sell_x_success = []
    sell_y_success = []
    sell_x_fail = []
    sell_y_fail = []

    for r in records:
        qty = r.get("quantity", 0.0)
        if qty <= 1e-8:
            continue

        if r["success"]:
            buy_x_success.append(r["buyer_price"])
            buy_y_success.append(qty)
            sell_x_success.append(r["seller_price"])
            sell_y_success.append(qty)
        else:
            buy_x_fail.append(r["buyer_price"])
            buy_y_fail.append(qty)
            sell_x_fail.append(r["seller_price"])
            sell_y_fail.append(qty)

    plt.figure(figsize=(8,6))
    plt.scatter(buy_x_success, buy_y_success, alpha=0.6, label="Buy Orders Executed")
    plt.scatter(buy_x_fail, buy_y_fail, alpha=0.4, label="Buy Orders Failed", marker="x")
    plt.scatter(sell_x_success, sell_y_success, alpha=0.6, label="Sell Orders Executed")
    plt.scatter(sell_x_fail, sell_y_fail, alpha=0.4, label="Sell Orders Failed", marker="x")

    title = f"Order Scatter for {good}"
    if round_idx is not None:
        title += f" at Round {round_idx + 1}"

    plt.xlabel("Proposed Price")
    plt.ylabel("Proposed Quantity")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_supply_demand_snapshot(market_history, good, round_idx):
    prices = market_history[good]["prices"][round_idx]
    demand = market_history[good]["demand_curves"][round_idx]
    supply = market_history[good]["supply_curves"][round_idx]
    executed_price = market_history[good]["executed_prices"][round_idx]

    plt.figure(figsize=(7,5))
    plt.plot(prices, demand, label="Demand")
    plt.plot(prices, supply, label="Supply")

    if np.isfinite(executed_price):
        plt.axvline(executed_price, linestyle="--", label=f"Executed Price = {executed_price:.2f}")

    plt.xlabel(f"Price of {good}")
    plt.ylabel("Quantity")
    plt.title(f"Supply and Demand for {good} at Round {round_idx + 1}")
    plt.legend()
    plt.show()

def plot_population(history):
    plt.figure(figsize=(6,4))
    plt.plot(history["population"])
    plt.xlabel("Time step")
    plt.ylabel("Population size")
    plt.title("Population Over Time")
    plt.show()

def plot_population_on_resource_grid(resource_grid, snapshot_history, round_idx, alpha_resources=0.75):
    if round_idx < 0 or round_idx >= len(snapshot_history):
        print(f"round_idx must be between 0 and {len(snapshot_history)-1}")
        return

    apple = resource_grid["apple"]
    barracuda = resource_grid["barracuda"]

    # Normalize for safety
    apple_norm = (apple - apple.min()) / (apple.max() - apple.min() + 1e-8)
    barracuda_norm = (barracuda - barracuda.min()) / (barracuda.max() - barracuda.min() + 1e-8)

    # RGB image:
    #   red   = 0
    #   green = apple
    #   blue  = barracuda
    rgb = np.zeros((apple.shape[0], apple.shape[1], 3))
    rgb[:, :, 1] = apple_norm
    rgb[:, :, 2] = barracuda_norm

    positions = snapshot_history[round_idx]["positions"]

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb, origin="lower", extent=[0, 1, 0, 1], alpha=alpha_resources)

    if len(positions) > 0:
        plt.scatter(
            positions[:, 0],
            positions[:, 1],
            c="red",
            s=10,
            alpha=0.8,
            label="Agents"
        )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Population on Resource Grid, Round {round_idx + 1}")
    plt.legend()
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
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2)
    plt.show()

# Instantiated Variables
snapshot_history = []
base_prices = {"apple": 2, "barracuda": 5}
price_history = {"apple": [], "barracuda": []}
grid_size = 50
raw_apple = np.random.rand(grid_size, grid_size)
resource_grid = {
    "apple": normalize(gaussian_filter(raw_apple, sigma=5)),
    "barracuda": normalize(gaussian_filter(1 - raw_apple, sigma=5))
}
trait_keys = ["intelligence", "wisdom", "strength", "dexterity", "charisma", "comeliness", "constitution", "metabolic_rate"]
num_agents = 1000
initial_food = 10*num_agents
market_history = {
    "apple": {
        "prices": [],
        "demand_curves": [],
        "supply_curves": [],
        "executed_prices": [],
        "executed_qty": []
    },
    "barracuda": {
        "prices": [],
        "demand_curves": [],
        "supply_curves": [],
        "executed_prices": [],
        "executed_qty": []
    }
}
order_history = {
    "apple": [],
    "barracuda": []
}
include_reasons = {"executed", "buyer_lacks_cash", "seller_lacks_inventory"}
num_rounds = 20

# Run
start = time.time()
agents, history, market_history, order_history, snapshot_history = population_evolution_market(num_agents, num_rounds, trait_keys, resource_grid, 0.1, market_history, order_history, snapshot_history)
end = time.time()
print("Time:", end - start, "seconds")
plot_population(history)
plot_traits(history)
plot_population_on_resource_grid(resource_grid, snapshot_history, 0)
plot_population_on_resource_grid(resource_grid, snapshot_history, 10)
plot_population_on_resource_grid(resource_grid, snapshot_history, 20)
plot_empirical_supply_demand_filtered(order_history, "apple", num_rounds, include_reasons=include_reasons)
plot_empirical_supply_demand_filtered(order_history, "barracuda", num_rounds, include_reasons=include_reasons)
plot_order_scatter(order_history, "apple", num_rounds)
plot_order_scatter(order_history, "barracuda", num_rounds)