from scipy.ndimage import convolve
from ant import Ant
import numpy as np
import random
from utils import *
import cv2

import matplotlib.pyplot as plt


class Environment:
    def __init__(self, map_shape, num_agents, diffusion_rate, evaporation_rate):
        self.map_shape = np.array(map_shape)
        self.diffusion_rate = diffusion_rate
        self.evaporation_rate = evaporation_rate

        self.blue_pheromone_map = np.zeros(map_shape)
        self.red_pheromone_map = np.zeros(map_shape)

        self.food_map = self.__generate_food_map(0, 2, 1000)
        self.ants = self.__generate_ants(num_agents)

    def __generate_ants(self, num_ants):
        ants = []
        directions = np.array(
            [[1, 0],
             [1, 1],
             [0, 1],
             [-1, 1],
             [-1, 0],
             [-1, -1],
             [0, -1],
             [1, -1]]
        )
        for _ in range(num_ants):
            random_direction = directions[np.random.choice(directions.shape[0]), :]
            ants.append(Ant(self, self.map_shape // 2, random_direction))
        return ants

    def __generate_food_map(self, number_food_sources, sigma, food_per_source):
        food_map = np.zeros(self.map_shape)

        for _ in range(number_food_sources):
            centre = (random.randint(0, self.map_shape[0]), random.randint(0, self.map_shape[1]))
            for _ in range(food_per_source):
                x = bound(random.gauss(centre[0], sigma), self.map_shape[0])
                y = bound(random.gauss(centre[1], sigma), self.map_shape[1])
                food_map[x, y] += 1
        return food_map

    def step(self):

        for ant in self.ants:
            ant.update_next_position()

        for ant in self.ants:
            ant.move_to_next_position()

        #self.diffuse_pheromones(self.food_map)

    def get_visual_map(self):
        visual_map = np.stack((self.blue_pheromone_map * 10, self.food_map * 10, self.red_pheromone_map * 10),
                              axis=2).astype(np.uint8)
        for ant in self.ants:
            visual_map[ant.position[0], ant.position[1], :] = 255

        return visual_map

    def diffuse_pheromones(self, array):
        current_multiplier = 1 - self.diffusion_rate
        orthogonal_multiplier = self.diffusion_rate / (4 + 2 ** 1.5)
        diagonal_multiplier = self.diffusion_rate / (4 * 2 ** 0.5 + 4)
        kernel = np.array([[diagonal_multiplier, orthogonal_multiplier, diagonal_multiplier],
                           [orthogonal_multiplier, current_multiplier, orthogonal_multiplier],
                           [diagonal_multiplier, orthogonal_multiplier, diagonal_multiplier]])
        kernel *= 1 - self.evaporation_rate

        """multiplier = diffusion_rate / 8
        kernel = np.array([[multiplier, multiplier, multiplier],
                           [multiplier, current_multiplier, multiplier],
                           [multiplier, multiplier, multiplier]])"""
        self.food_map = convolve(array, kernel, mode="constant")

