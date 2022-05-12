from ant import Ant
from ant_states import AntStates
from utils import *


class Environment:
    def __init__(self, map, num_agents, diffusion_rate, evaporation_rate):
        """
        Initialises the environment
        :param map: The map of the environment
        :param num_agents: The number of ants in the environment
        :param diffusion_rate: The rate pheromones diffuse
        :param evaporation_rate: The rate pheromones evaporate
        """
        assert 0 <= diffusion_rate <= 1, "Diffusion rate must be in [0..1]"
        assert 0 <= evaporation_rate <= 1, "Evaporation rate must be in [0..1]"

        self.map = map

        self.diffusion_rate = diffusion_rate
        self.evaporation_rate = evaporation_rate

        self.ants = self.__generate_ants(num_agents)

    def __generate_ants(self, num_ants):
        """
        Generates all ants at the colony location, facing them in a random direction
        :param num_ants: the number of ants to place in the environment
        :return: None
        """
        ants = []
        for _ in range(num_ants):
            random_direction = self.map.directions[np.random.choice(self.map.directions.shape[0]), :]
            ants.append(Ant(self.map, self.map.colony_position, random_direction))
        return ants

    def step(self):
        """
        Execute one timestep of the environment
        Ants must calculate where to move all at the same time, then execute the movement at the same time. Otherwise
        the ants that move first will leave pheromones behind which will impact the actions of the other ants
        :return: None
        """

        for ant in self.ants:
            ant.update_next_position()

        for ant in self.ants:
            ant.move_to_next_position()

        self.map.diffuse_pheromones(self.diffusion_rate, self.evaporation_rate)

    def visualise(self):
        """
        Returns an image to visualise the environment
        :return:
        """
        visual_map = self.map.visualise()
        for ant in self.ants:
            # ants searching for food are coloured white
            if ant.status == AntStates.SEARCHING_FOR_FOOD:
                visual_map[ant.position[0], ant.position[1], :] = 255
            # ants returning with food are coloured green
            elif ant.status == AntStates.RETURNING_WITH_FOOD:
                visual_map[ant.position[0], ant.position[1], :] = np.array([0, 255, 0])
            # exhausted ants, ants returning without food are coloured aqua
            else:
                visual_map[ant.position[0], ant.position[1]] = np.array([0, 255, 255])

        return visual_map.astype(np.uint8)
