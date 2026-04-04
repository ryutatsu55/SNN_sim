import numpy as np
from pydantic import BaseModel, Field, ValidationError
from src.core.registry import DATA_LOADERS

@DATA_LOADERS.register("pqn_test")
class SpatialDataLoader:
    def __init__(self, config: dict):
        self.config = config

    def load_data(self):
        tmax = self.config["simulation"]["duration"]
        input_current = 0.12
        number_of_iterations = int(tmax / self.config["simulation"]["dt"])
        I_in = np.zeros((number_of_iterations, self.config["simulation"]["N"]))
        I_in[int(number_of_iterations/4):int(number_of_iterations/4*3), :] = input_current
        return I_in