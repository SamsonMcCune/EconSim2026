from Agent_Class import Agent
from Agent_Class import Grid
from Relationship_Class import Relationship
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)
grid = Grid(1,1,20)

def first_generation(num_agents):
    agents = [Agent(i) for i in range(num_agents)]
    positions = np.array([agent.position for agent in agents])
    cell_x, cell_y = grid.assign_cells(positions)
    return agents, positions, cell_x, cell_y

def build_neighbors(positions, cell_x, cell_y, agents):
    relationships = {}
    n = len(cell_x)
    grid_lookup = {}
    for i in range(n):
        key = (cell_x[i], cell_y[i])
        grid_lookup.setdefault(key, []).append(i)
    for i in range(n):
        cx = cell_x[i]
        cy = cell_y[i]
        x_i, y_i = positions[i]
        r_i = agents[i].radius
        cell_radius = min(2, int(np.ceil(r_i / grid.cell_size_x)))
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                neighbor_cell = (cx + dx, cy + dy)
                if neighbor_cell not in grid_lookup:
                    continue
                for j in grid_lookup[neighbor_cell]:
                    if j <= i:
                        continue
                    x_j, y_j = positions[j]
                    dx_ = x_i - x_j
                    dy_ = y_i - y_j
                    dist2 = dx_*dx_ + dy_*dy_
                    r_sum = agents[i].radius + agents[j].radius
                    if dist2 <= r_sum * r_sum:
                        relationship_i_j = Relationship(agents[i], agents[j])
                        relationships[relationship_i_j.key] = relationship_i_j
    return relationships

def build_friendships(relationships):
    return [r for r in relationships.values() if r.friends]
    

def visualize_network(positions, relationships, subdivisions):
    plt.figure(figsize=(6,6))
    plt.scatter(positions[:, 0], positions[:, 1], zorder=2)
    for rel in relationships.values():
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
    plt.title("Agent Network (Friendships)")
    plt.show()

agents, positions, cell_x, cell_y = first_generation(2000)
relationships = build_neighbors(positions, cell_x, cell_y, agents)
friendships = build_friendships(relationships)
print(f"There are {len(friendships)} frienships.")
visualize_network(positions, relationships, grid.x_y_subdivisions)