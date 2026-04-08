import numpy as np

# Structure: there is a buyer and there is a seller.
# The buyer has a set of objects that it can use to purchase from the seller.
# The seller has a set of objects that it can sell to the buyer.
# Both are maximizing their total utility.
# Cash is a good and is valued equally by both parties.

np.random.seed(2)

buyer_initial_wealth = 0
n_b = 0
m_b = 0
o_b = 0
p_b = 0
cash_b = 0
w_b = 100

seller_initial_wealth = 0
n_s = 0
m_s = 0
o_s = 0
p_s = 0
cash_s = 0
w_s = 100


buyer_wealth = [buyer_initial_wealth, n_b, m_b, o_b, p_b, cash_b]
for i in range(len(buyer_wealth) - 1):
    buyer_wealth[i] = np.random.randint(0, w_b + 1)
    w_b = buyer_wealth[i]
buyer_wealth[5] = int(buyer_wealth[0] - np.sum(buyer_wealth[1:4]))

seller_wealth = [seller_initial_wealth, n_s, m_s, o_s, p_s, cash_s]
for i in range(len(seller_wealth) - 1):
    seller_wealth[i] = np.random.randint(0, w_s + 1)
    w_s = seller_wealth[i]
seller_wealth[5] = int(seller_wealth[0] - np.sum(seller_wealth[1:4]))


buyer_inventory = {
    "apples": buyer_wealth[1],
    "bananas": buyer_wealth[2],
    "cows": buyer_wealth[3],
    "dandelions": buyer_wealth[4],
    "money": buyer_wealth[5]
}

seller_inventory = {
    "apples": seller_wealth[1],
    "bananas": seller_wealth[2],
    "cows": seller_wealth[3],
    "dandelions": seller_wealth[4],
    "money": seller_wealth[5]
}


print(f"The buyer starts with {buyer_wealth[0]} items to be allocated into {buyer_inventory}")
print(f"The seller starts with {seller_wealth[0]} items to be allocated into {seller_inventory}")