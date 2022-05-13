import random
from utils import *
from ant_states import AntStates


class Ant:
    def __init__(self, map, starting_position, direction):
        """
        Initialises an Ant
        :param map: The map which the ant is on
        :param starting_position: A numpy array indicating the starting location
        :param direction: A numpy array containing a vector towards which the ant is initially facing
        """
        self.next_position = None
        self.map = map
        self.direction = direction
        self.position = starting_position

        self.status = AntStates.SEARCHING_FOR_FOOD

        # the chance an ant will make a random action
        self.random_movement_chance = 0.05

        # the ant drops some percent of his pheromones each time step.
        self.initial_pheromone_bank = 10000
        self.pheromone_drop_percent = 0.01

        # how long until the ant stops dropping pheromones. ie, time until the ant is exhausted and wants to come home
        self.initial_pheromone_deplete_time = 100

        self.pheromone_bank = self.initial_pheromone_bank
        self.pheromone_deplete_time = self.initial_pheromone_deplete_time

    def __drop_pheromone(self):
        """
        Drops pheromone at the ant's current position
        :return: None
        """
        if self.pheromone_deplete_time > 0:

            # decrease pheromone level in the pheromone_bank
            pheromone_drop_amount = self.pheromone_bank * self.pheromone_drop_percent
            self.pheromone_bank -= pheromone_drop_amount
            self.pheromone_deplete_time -= 1

            # deposit the pheromone on the map
            if self.status == AntStates.SEARCHING_FOR_FOOD:
                self.map.blue_pheromone_map[self.position[0], self.position[1]] += pheromone_drop_amount
            elif self.status == AntStates.RETURNING_WITH_FOOD:
                self.map.red_pheromone_map[self.position[0], self.position[1]] += pheromone_drop_amount

    def __reset_pheromones(self):
        """
        Resets the pheromone attributes to their initial values
        :return:
        """
        self.pheromone_bank = self.initial_pheromone_bank
        self.pheromone_deplete_time = self.initial_pheromone_deplete_time

    def update_next_position(self):
        """
        Updates self.next_position to determine where the ant should move next step
        :return: None
        """
        # if the ant is not returning with food, and they are currently at food
        if not self.status == AntStates.RETURNING_WITH_FOOD and self.map.food_map[self.position[0], self.position[1]]:
                self.map.remove_food(self.position)
                self.status = AntStates.RETURNING_WITH_FOOD
                self.__reset_pheromones()
                self.__orientate()
        # if the ant is not searching for food, and they are inside the colony
        elif not self.status == AntStates.SEARCHING_FOR_FOOD and self.map.is_close_to_colony(self.position):
            # if the any is at the centre of the colony
            if np.array_equal(self.position, self.map.colony_position):
                self.__reset_pheromones()
                self.status = AntStates.SEARCHING_FOR_FOOD
                self.next_position = self.position + self.direction
            else:
                self.__point_towards(self.map.colony_position)
        # if the ant is searching for food but can't find any
        elif self.status == AntStates.SEARCHING_FOR_FOOD and self.pheromone_deplete_time == 0:
            self.status = AntStates.RETURNING_WITHOUT_FOOD
            self.__orientate()
        # if food is 1 step away, go to food, else navigate using pheromones
        else:
            # get 3 points in front of the ant
            if np.linalg.norm(self.direction) == 1:
                sample_points = np.array([self.position + (self.direction + normal(self.direction)),
                                          self.position + self.direction,
                                          self.position + (self.direction - normal(self.direction))], dtype=np.int16)
            else:
                sample_points = np.array([self.position + (self.direction + normal(self.direction)) / 2,
                                          self.position + self.direction,
                                          self.position + (self.direction - normal(self.direction)) / 2], dtype=np.int16)

            pheromone_map = self.__pheromone_map()

            pheromones = []
            points = []
            food_points = []
            # get pheromone levels at each sample_point, and check if any of the sample points contain food
            for point in sample_points:
                # if the point isn't impassable
                if not self.map.impassable_map[point[0], point[1]]:
                    # if food is at the point
                    if self.map.food_map[point[0], point[1]]:
                        food_points.append(point)
                    points.append(point)
                    pheromones.append(pheromone_map[point[0], point[1]])

            # if there are points which the ant can access
            if len(points) > 0:
                # if there is food nearby and the ant isn't already carrying food
                if not self.status == AntStates.RETURNING_WITH_FOOD and len(food_points) > 0:
                    self.next_position = food_points[np.random.choice(len(food_points))]
                else:
                    if self.status == AntStates.SEARCHING_FOR_FOOD:
                        # act randomly
                        if random.random() < self.random_movement_chance:
                            self.next_position = points[np.random.choice(len(points))]
                        # act in accordance with the softmax of the pheromone levels.
                        # this is admittedly arbitrary, as the levels of pheromones will differ wildly as other params
                        # change.
                        else:
                            probability = softmax(pheromones)
                            self.next_position = points[np.random.choice(len(points), p=probability)]
                    else:
                        # select the action with the highest concentration of pheromones
                        probability = one_hot(pheromones)
                        self.next_position = points[np.random.choice(len(points), p=probability)]
            # no valid moves to make
            else:
                self.next_position = self.position

    def move_to_next_position(self):
        """
        Move the ant to it's next position
        :return: None
        """
        self.__drop_pheromone()

        # if the ant is not moving, randomly change its direction
        if np.array_equal(self.position, self.next_position):
            self.direction = self.map.directions[np.random.choice(self.map.directions.shape[0]), :]
        # update direction vector
        else:
            self.direction = self.next_position - self.position

        self.position = self.next_position

    def __orientate(self):
        """
        Points ant in the direction of the highest pheromone levels in all 8 directions
        :return:
        """
        sample_points = []
        for direction in self.map.directions:
            sample_points.append(self.position + direction)

        points = []
        pheromones = []
        for point in sample_points:
            if not self.map.impassable_map[point[0], point[1]]:
                points.append(point)
                pheromones.append(self.__pheromone_map()[point[0], point[1]])

        probability = one_hot(pheromones)
        self.next_position = points[np.random.choice(len(points), p=probability)]

    def __pheromone_map(self):
        """
        Gets the pheromone map the ant is currently using to navigate
        :return:
        """
        if self.status == AntStates.SEARCHING_FOR_FOOD:
            pheromone_map = self.map.red_pheromone_map
        else:
            pheromone_map = self.map.blue_pheromone_map

        return pheromone_map

    def __point_towards(self, position):
        """
        Points the ant towards a position
        :param position: The position to be pointed towards
        :return: None
        """
        direction = (position - self.position)
        unit_direction = (direction / np.linalg.norm(direction)).round().astype(int)
        self.next_position = self.position + unit_direction
