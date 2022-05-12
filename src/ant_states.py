from enum import Enum, auto


class AntStates(Enum):
    """
    These are to be applied to ants.

    SEARCHING_FOR_FOOD: Navigates using red pheromones and lays blue pheromones behind
    RETURNING_WITH_FOOD: Navigates using blue pheromones and lays red pheromones behind. Carrying food
    RETURNING_WITHOUT_FOOD: Navigates using red pheromones and lays no pheromones behind
    """
    SEARCHING_FOR_FOOD = "SEARCHING_FOR_FOOD",
    RETURNING_WITH_FOOD = "RETURNING_WITH_FOOD",
    RETURNING_WITHOUT_FOOD = "RETURNING_WITHOUT_FOOD"
