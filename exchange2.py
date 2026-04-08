import numpy as np
import matplotlib.pyplot as plt
from agent import Agent

base_prices = {
    "apple": 2,
    "barracuda": 5
}

price_history = {
    "apple": [],
    "barracuda": []
}

def agent_learning(agents):
    for agent in agents:
        agent.learning()
    return agents

def market_generation(num_agents):
    agents = np.empty(num_agents, dtype=object)
    for i in range(num_agents):
        agents[i] = Agent(f"{i}", np.random.randint(0, 100))
    return agents

def transaction_cost(agent_1, agent_2, N):
    t_cost = 1/N*abs(agent_1 - agent_2)
    return t_cost

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
        #print(f"Agent {buyer.name} bought {trade_qty} {good} from agent {seller.name} for ${price:.2f}")
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

def gini(wealth):
    wealth = np.sort(wealth)
    n = len(wealth)
    cumulative = np.cumsum(wealth)
    return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n

def plot_price_histograms(price_history, bins=50):
    for good, prices in price_history.items():
        if len(prices) == 0:
            print(f"No trades recorded for {good}.")
            continue
        plt.figure()
        plt.hist(prices, bins=bins)
        plt.title(f"{good.capitalize()} transaction prices (histogram)")
        plt.xlabel("Price")
        plt.ylabel("Count")
        plt.axvline(np.mean(prices), linestyle="--", label=f"mean = {np.mean(prices):.3f}")
        plt.legend()
        plt.show()

def plot_price_timeseries(price_history):
    for good, prices in price_history.items():
        if len(prices) == 0:
            print(f"No trades recorded for {good}.")
            continue

        prices = np.array(prices, dtype=float)
        plt.figure()
        plt.plot(prices, label="price")

        plt.title(f"{good.capitalize()} price over trades")
        plt.xlabel("Trade index")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

def get_total_wealth(agents, apple_price, barracuda_price):
    return np.array([agent.inventory[2] + apple_price * agent.inventory[0] + barracuda_price * agent.inventory[1] for agent in agents], dtype=float)

def plot_total_wealth_distribution(agents, apple_price, barracuda_price, bins=50):
    if apple_price is None or barracuda_price is None:
        print("Cannot compute total wealth: missing market prices.")
        return

    w = get_total_wealth(agents, apple_price, barracuda_price)
    plt.figure()
    plt.hist(w, bins=bins)
    plt.title("Total wealth distribution (cash + priced goods)")
    plt.xlabel("Wealth")
    plt.ylabel("Count")
    plt.axvline(np.mean(w), linestyle="--", label=f"mean = {np.mean(w):.2f}")
    plt.legend()
    plt.show()

def plot_inventory_distributions(agents, bins=50):
    apples = np.array([a.inventory[0] for a in agents], dtype=float)
    barrs  = np.array([a.inventory[1] for a in agents], dtype=float)

    plt.figure()
    plt.hist(apples, bins=bins)
    plt.title("Apple holdings distribution")
    plt.xlabel("Apples")
    plt.ylabel("Count")
    plt.show()

    plt.figure()
    plt.hist(barrs, bins=bins)
    plt.title("Barracuda holdings distribution")
    plt.xlabel("Barracudas")
    plt.ylabel("Count")
    plt.show()

def plot_lorenz_curve(wealth):

    wealth = np.array(wealth)
    wealth = np.sort(wealth)

    n = len(wealth)

    cumulative_wealth = np.cumsum(wealth)
    total_wealth = cumulative_wealth[-1]

    lorenz_y = np.insert(cumulative_wealth / total_wealth, 0, 0)
    lorenz_x = np.linspace(0, 1, n + 1)

    gini_coefficient = gini(wealth)

    plt.figure()
    plt.title(f"Lorenz Curve (Gini = {gini_coefficient:.3f})")
    plt.plot(lorenz_x, lorenz_y, label="Lorenz Curve")
    plt.plot([0,1], [0,1], linestyle="--", label="Perfect Equality")
    plt.xlabel("Cumulative Share of Agents")
    plt.ylabel("Cumulative Share of Wealth")
    plt.legend()
    plt.show()

def plot_gini_over_time(gini_values):

    gini_values = np.array(gini_values)
    periods = np.arange(len(gini_values))

    plt.figure()
    plt.plot(periods, gini_values, linewidth=2)

    plt.xlabel("Period")
    plt.ylabel("Gini Coefficient")
    plt.title("Evolution of Wealth Inequality Over Time")

    plt.ylim(0,1)
    plt.grid(alpha=0.3)

    plt.show()

# Does this even work? Does not converge to 0 at all.
def mrs_dispersion(agents):
    mrs_values = [agent.mrs()[0] for agent in agents]
    return np.std(mrs_values)

interest_rate = .24
delta = 1
periods = 100
rounds = 1000
num_agents = 1000
agents = market_generation(num_agents)
agents, gini_rounds = bilateral_market(agents, periods, rounds, delta)
successful_trades = sum(len(v) for v in price_history.values())
apple_price = price_history["apple"][-1] if price_history["apple"] else None
barracuda_price = price_history["barracuda"][-1] if price_history["barracuda"] else None
gini_final = gini(get_total_wealth(agents, apple_price, barracuda_price))
mrs_output = mrs_dispersion(agents)
print(f"The MRS Output is {mrs_output}")
print(f"Final gini coefficient is {gini_final}")
plot_gini_over_time(gini_rounds)