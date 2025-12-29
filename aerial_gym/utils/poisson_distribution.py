import numpy as np
import random
import math

class PoissonDiscSampler:
    def __init__(self, width, height, radius, seed=None):
        self.width = width
        self.height = height
        self.radius = radius
        self.radius2 = radius ** 2
        self.cell_size = radius / math.sqrt(2)
        self.grid_width = math.ceil(width / self.cell_size)
        self.grid_height = math.ceil(height / self.cell_size)
        self.grid = np.full((self.grid_width, self.grid_height), None, dtype=object)  # Store as object for flexibility
        self.active_samples = []
        if seed is not None:
            np.random.seed(seed)

    def add_sample(self, sample):
        self.active_samples.append(sample)
        grid_x = int(sample[0] / self.cell_size)
        grid_y = int(sample[1] / self.cell_size)
        self.grid[grid_x, grid_y] = sample
        return sample

    def is_far_enough(self, sample):
        grid_x = int(sample[0] / self.cell_size)
        grid_y = int(sample[1] / self.cell_size)
        x_min = max(grid_x - 2, 0)
        y_min = max(grid_y - 2, 0)
        x_max = min(grid_x + 2, self.grid_width - 1)
        y_max = min(grid_y + 2, self.grid_height - 1)

        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                s = self.grid[x, y]
                if s is not None:
                    d = np.array(s) - np.array(sample)
                    if np.dot(d, d) < self.radius2:
                        return False
        return True

    def samples(self):
        # First sample
        first_sample = (random.uniform(0, self.width), random.uniform(0, self.height))
        yield self.add_sample(first_sample)

        while self.active_samples:
            i = random.randint(0, len(self.active_samples) - 1)
            sample = self.active_samples[i]
            found = False

            for _ in range(30):  # k attempts
                angle = random.uniform(0, 2 * math.pi)
                r = math.sqrt(random.uniform(self.radius2, 4 * self.radius2))
                candidate = sample + r * np.array([math.cos(angle), math.sin(angle)])

                if 0 <= candidate[0] < self.width and 0 <= candidate[1] < self.height and self.is_far_enough(candidate):
                    found = True
                    yield self.add_sample(candidate)
                    break

            if not found:
                self.active_samples.pop(i)
