import numpy as np

apples_cost = 2
barracudas_cost = 5
cash_cost = 1
lr = .015
apples = 0
barracudas = 0
cash = 0

def dictionary_to_vector(traits, keys):
    return np.array([traits[k] for k in keys])

def softmax(w):
    exp = np.exp(w - np.max(w))
    return exp / np.sum(exp)

trait_keys = ["intelligence", "wisdom", "strength", "dexterity", "charisma", "comeliness", "constitution"]

class Agent:

    def __init__(self, name, generation):
        self.id = name
        self.generation = generation
        self.position = self.random_uniform_position()
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
        self.radius = 0.02 + 0.08*(self.genome["dexterity"]/200)
        self.relationships = {}
        self.genome_vector = dictionary_to_vector(self.genome, trait_keys)
        self.genome_vector_normalized = (self.genome_vector - 1)/np.linalg.norm(self.genome_vector, keepdims=True)
        self.income = self.genome["intelligence"]*np.random.randint(1,5)
        self.w = np.array([1.0, 1.0, 1.0])
        self.true_CD_exponents = softmax(np.random.rand(3))
        self.expected_CD_exponents = softmax(self.w)
        self.inventory = np.array(self.random_allocation(), dtype=float)
        self.saving_rate = np.random.rand()

    def random_uniform_position(self):
        position = np.random.rand(2)
        return position
    
    def random_normal_position(self, mean, variance):
        position = (np.random.normal(mean, variance, 2))
        return position
    
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
    
    def mrs(self):
        alpha, beta, gamma = self.expected_CD_exponents
        apples, barracudas, cash = self.inventory
        mrs_ac = (alpha/gamma) * (cash / apples)
        mrs_bc = (beta/gamma) * (cash / barracudas)
        return mrs_ac, mrs_bc
    

