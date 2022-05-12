import numpy as np
from scipy.ndimage import convolve
import random
from utils import *


class Map:
    def __init__(self, map_shape, n_food_sources=1, food_sigma=2, food_per_source=1000, colony_size=5):
        """
        Initialises the map
        :param map_shape: 2d array like
        :param n_food_sources: number of food sources to place in the environment
        :param food_sigma: how spread out the food sources will be
        :param food_per_source: how many food units will be in each food source
        :param colony_size: the physical size of the colony
        """
        self.map_shape = np.array(map_shape)

        # stores array of directions, where each direction is a vector
        self.directions = np.array(
            [[1, 0],
             [1, 1],
             [0, 1],
             [-1, 1],
             [-1, 0],
             [-1, -1],
             [0, -1],
             [1, -1]]
        )

        # blue pheromones are laid when leaving the colony
        self.blue_pheromone_map = np.zeros(map_shape)
        # red pheromones are laid when returning form the colony
        self.red_pheromone_map = np.zeros(map_shape)

        # food map stores all food sources
        self.food_remaining = 0
        self.food_map = self.__generate_food_map(n_food_sources, food_sigma, food_per_source)

        # impassable_map stores locations of impassable positions.
        # pheromones will diffuse through impassable positions, and ants can move diagonally over impassable positions
        self.impassable_map = np.ones(map_shape)
        self.impassable_map[1:-1, 1:-1] = 0

        # colony placed in the centre of the map
        self.colony_position = np.array(map_shape) // 2

        self.colony_size = colony_size

        # stores which positions are inside the colony
        self.colony_map = np.zeros(map_shape)
        for point in np.ndindex(map_shape):
            self.colony_map[point[0], point[1]] = self.is_close_to_colony(point)

    def __generate_food_map(self, number_food_sources, sigma, food_per_source):
        """
        Generates food sources randomly using a 2d gaussian
        :param number_food_sources: number of food sources in the map
        :param sigma: the spread of the food sources
        :param food_per_source: the number of food units per food source
        :return: 2d array of ints, representing the amount of food in each position
        """
        food_map = np.zeros(self.map_shape)

        for _ in range(number_food_sources):
            centre = (random.randint(0, self.map_shape[0]), random.randint(0, self.map_shape[1]))
            for _ in range(food_per_source):
                x = bound(random.gauss(centre[0], sigma), self.map_shape[0])
                y = bound(random.gauss(centre[1], sigma), self.map_shape[1])
                food_map[x, y] += 1
                self.food_remaining += 1
        return food_map

    def visualise(self):
        """
        Returns an image representing a visualisation of the map
        :return:
        """
        visual_map = np.stack((self.red_pheromone_map * 0.1, self.food_map * 10, self.blue_pheromone_map * 0.1),
                              axis=2).astype(np.uint8)

        # impassable positions are grey
        visual_map[self.impassable_map == 1, :] = 125
        # the colony is brown
        visual_map[self.colony_map == 1] = np.array([139, 69, 19])

        return visual_map

    def diffuse_pheromones(self, diffusion_rate, evaporation_rate):
        """
        Diffuses pheromones outwards using a kernel, then evaporates them
        :param diffusion_rate: the rate pheromones diffuse
        :param evaporation_rate: the rate pheromones evaporate
        :return: None
        """
        current_multiplier = 1 - diffusion_rate

        # the kernel empirically gives a better circle for small numbers of steps
        orthogonal_multiplier = diffusion_rate / (4 + 2 ** 1.5)
        diagonal_multiplier = diffusion_rate / (4 * 2 ** 0.5 + 4)
        kernel = np.array([[diagonal_multiplier, orthogonal_multiplier, diagonal_multiplier],
                           [orthogonal_multiplier, current_multiplier, orthogonal_multiplier],
                           [diagonal_multiplier, orthogonal_multiplier, diagonal_multiplier]])

        # this kernel in the limit will result in a circle, but empirically is not a circle for a small number of steps
        """multiplier = diffusion_rate / 8
        kernel = np.array([[multiplier, multiplier, multiplier],
                           [multiplier, current_multiplier, multiplier],
                           [multiplier, multiplier, multiplier]])"""

        # evaporate pheromones
        kernel *= 1 - evaporation_rate

        self.blue_pheromone_map = convolve(self.blue_pheromone_map, kernel)
        self.red_pheromone_map = convolve(self.red_pheromone_map, kernel)

    def is_close_to_colony(self, position):
        """
        Returns True if a position is inside the colony, false otherwise
        :param position: The position to find the distance to the colony
        :return: True if the position is inside the colony, false otherwise
        """
        direction = self.colony_position - position
        distance = np.linalg.norm(direction)

        return distance <= self.colony_size

    def remove_food(self, position):
        """
        Remove food from a position
        :param position: The position to remove food from
        :return: None
        """
        if self.food_map[position[0], position[1]] > 0:
            self.food_map[position[0], position[1]] -= 1
            self.food_remaining -= 1
