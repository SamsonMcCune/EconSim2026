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