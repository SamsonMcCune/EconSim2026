import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PopulationEnv(gym.Env):
    def __init__(self, num_agents, trait_keys):
        super().__init__()

        self.num_agents = num_agents
        self.trait_keys = trait_keys

        # Action: control signal (continuous)
        self.action_space = spaces.Box(
            low=np.array([0.5]),
            high=np.array([2.0]),
            dtype=np.float32
        )

        # State: 5 features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agents, _ = first_generation(self.num_agents)
        self.next_id = self.num_agents
        self.t = 0

        self.prev_population = len(self.agents)
        self.prev_growth = 0

        return self._get_state(), {}

    def step(self, action):
        control_signal = float(action[0])

        # --- Simulation step ---
        relationships = build_relationships_kdtree(self.agents)
        friendships = build_friendships(relationships)
        marriages = build_marriages(friendships)

        children, self.next_id = reproduce(
            marriages, self.trait_keys, self.next_id, 0.4
        )

        self.agents.extend(children)

        self.agents = [
            a for a in self.agents
            if survives(a, self.t, len(self.agents), control_signal)
        ]

        # --- Metrics ---
        current_population = len(self.agents)

        growth = (current_population - self.prev_population) / max(self.prev_population, 1)
        growth_accel = growth - self.prev_growth

        # --- Reward ---
        reward = -((growth - 0.03) ** 2)

        # penalties
        if current_population < 50:
            reward -= 1
        if current_population > 500000:
            reward -= 1

        # --- Update state ---
        self.prev_population = current_population
        self.prev_growth = growth
        self.t += 1

        done = self.t >= 100

        return self._get_state(), reward, done, False, {}

    def _get_state(self):
        pop = len(self.agents)

        pop_norm = pop / 10000  # normalization

        avg_const = np.mean([a.genome["constitution"] for a in self.agents]) if self.agents else 0
        avg_age = np.mean([self.t - a.generation for a in self.agents]) if self.agents else 0

        growth = self.prev_growth
        growth_accel = 0  # already included in step logic

        return np.array([
            pop_norm,
            growth,
            growth_accel,
            avg_const / 100,
            avg_age / 100
        ], dtype=np.float32)