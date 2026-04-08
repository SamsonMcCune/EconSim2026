import numpy as np
np.random.seed(1)
apples_cost = 2
barracudas_cost = 5
cash_cost = 1
lr = .015
apples = 0
barracudas = 0
cash = 0

def softmax(w):
    exp = np.exp(w - np.max(w))
    return exp / np.sum(exp)

class Agent:

    def __init__(self, name, income):
        
        # First name is for individual labeling.
        # Last name is for familial/regional labeling
        # Gender is for reproduction.
        # Intelligence is risk preference.
        # Wisdom is time preference.
        # Strength is for inventory size
        # Dexterity is for transportation costs
        # Charisma is for bargaining power
        # Comeliness is for relationship development
        # Constitution is for one's willingness to accept negative things, sort of like fortitude. Could be removed.

        self.name = {
            "first name": self.first_name,
            "last name": self.last_name,
        }

        self.genome = {
            "gender": self.gender,
            "intelligence": self.IQ,
            "wisdom": self.wisdom,
            "strength": self.strength,
            "dexterity": self.dexterity,
            "charisma": self.charisma,
            "comeliness": self.comeliness,
            "constitution": self.constitution,
        }

        self.name = name
        self.income = income
        self.w = np.array([1.0, 1.0, 1.0])
        self.true_CD_exponents = softmax(np.random.rand(3))
        self.expected_CD_exponents = softmax(self.w)
        self.inventory = np.array(self.random_allocation(), dtype=float)
        self.saving_rate = np.random.rand()

    def log_true_utility(self, bundle):
        a, b, c = np.maximum(bundle, 1e-8)
        alpha, beta, gamma = self.true_CD_exponents
        return alpha*np.log(a) + beta*np.log(b) + gamma*np.log(c)
    
    def log_expected_utility(self, bundle):
        a, b, c = np.maximum(bundle, 1e-8)
        alpha, beta, gamma = self.expected_CD_exponents
        return alpha*np.log(a) + beta*np.log(b) + gamma*np.log(c)


    def update_beliefs(self, bundle):
        a, b, c = bundle
        logs = np.array([np.log(a), np.log(b), np.log(c)])
        true_log = self.log_true_utility(bundle)
        pred_log = self.log_expected_utility(bundle)
        error = true_log - pred_log
        alpha_hat = self.expected_CD_exponents
        gradient = error * alpha_hat * (logs - pred_log)
        self.w += lr * gradient
        self.expected_CD_exponents = softmax(self.w)

    def learning(self, rounds=10):
        for r in range(rounds):
            apples_owned, barracudas_owned, cash_owned = self.random_allocation()
            bundle_owned = apples_owned, barracudas_owned, cash_owned 
            self.update_beliefs(bundle_owned)
        return self.expected_CD_exponents

    def random_allocation(self):
        apples_spread = 0.1 + 3.9*np.random.rand()
        apples_allocation = self.income / apples_spread
        barracudas_allocation = self.income - apples_allocation
        apples_owned = apples_allocation // apples_cost
        barracudas_owned = barracudas_allocation // barracudas_cost
        cash_owned = self.income - (apples_owned*apples_cost + barracudas_owned*barracudas_cost)
        apples_owned = max(1e-6, apples_owned)
        barracudas_owned = max(1e-6, barracudas_owned)
        cash_owned = max(1e-6, cash_owned)
        return apples_owned, barracudas_owned, cash_owned
    
    def optimal_bundle(self):
        alpha, beta, gamma = self.expected_CD_exponents
        apples_float = alpha * self.income / apples_cost
        barracudas_float = beta * self.income / barracudas_cost
        apples = int(np.floor(apples_float))
        barracudas = int(np.floor(barracudas_float))
        cash = self.income - (apples * apples_cost + barracudas * barracudas_cost)
        bundle = [apples, barracudas, cash]
        return bundle
    
    def mrs(self):
        alpha, beta, gamma = self.expected_CD_exponents
        apples, barracudas, cash = self.inventory
        mrs_ac = (alpha/gamma) * (cash / apples)
        mrs_bc = (beta/gamma) * (cash / barracudas)
        return mrs_ac, mrs_bc
    
    def saving_appreciation(self, interest_rate):
        cash = self.inventory[2]
        saved = cash * self.saving_rate
        interest = saved * interest_rate
        self.inventory[2] += interest
        return interest
    
    def consume(self, depreciation_rate_apples, depreciation_rate_barracudas):
        consume_apples = 2
        consume_barracudas = 1
        utility = self.log_true_utility([consume_apples, consume_barracudas])
        self.inventory[0] -= consume_apples
        self.inventory[1] -= consume_barracudas
        depreciation_apples = np.floor(depreciation_rate_apples*self.inventory[0])
        depreciation_barracudas = np.floor(depreciation_rate_barracudas*self.inventory[1])
        self.inventory[0] -= depreciation_apples
        self.inventory[1] -= depreciation_barracudas
        return utility

    def production():
        return