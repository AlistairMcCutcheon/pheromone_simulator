import numpy as np
from utils import *


class Ant:
    def __init__(self, env, starting_position, direction):
        self.next_position = None
        self.env = env
        self.direction = direction
        self.position = starting_position
        self.searching_for_food = True

    def update_next_position(self):
        if np.linalg.norm(self.direction) == 1:
            sample_points = np.array([self.position + (self.direction + normal(self.direction)),
                                      self.position + self.direction,
                                      self.position + (self.direction - normal(self.direction))], dtype=np.int16)
        else:
            sample_points = np.array([self.position + (self.direction + normal(self.direction)) / 2,
                                      self.position + self.direction,
                                      self.position + (self.direction - normal(self.direction)) / 2], dtype=np.int16)

        if self.searching_for_food:
            pheromone_map = self.env.red_pheromone_map
        else:
            pheromone_map = self.env.blue_pheromone_map

        pheromones = []
        for point in sample_points:
            pheromones.append(pheromone_map[point[0], point[1]])
        probability = softmax(pheromones)

        self.next_position = sample_points[np.random.choice(sample_points.shape[0], p=probability)]

    def move_to_next_position(self):
        if self.searching_for_food:
            self.env.blue_pheromone_map[self.position[0], self.position[1]] += 1
        else:
            self.env.red_pheromone_map[self.position[0], self.position[1]] += 1

        self.direction = self.next_position - self.position

        self.position = self.next_position
