import numpy as np

def dictionary_to_vector(traits, keys):
    return np.array([traits[k] for k in keys])

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

    def random_uniform_position(self):
        position = np.random.rand(2)
        return position
    
    def random_normal_position(self, mean, variance):
        position = (np.random.normal(mean, variance, 2))
        return position
    
class Grid:
    def __init__(self, grid_size_x, grid_size_y, x_y_subdivisions):
        self.x_y_subdivisions = x_y_subdivisions
        self.cell_size_x = grid_size_x / self.x_y_subdivisions
        self.cell_size_y = grid_size_y / self.x_y_subdivisions

    def assign_cells(self, positions):
        cell_x = np.floor(positions[:, 0] / self.cell_size_x).astype(int)
        cell_y = np.floor(positions[:, 1] / self.cell_size_y).astype(int)
        cell_x = np.clip(cell_x, 0, self.x_y_subdivisions - 1)
        cell_y = np.clip(cell_y, 0, self.x_y_subdivisions - 1)
        return cell_x, cell_y