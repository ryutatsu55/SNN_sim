# src/models/neurons/lif.py
from models.neurons.base_models import BaseNeuron
import numpy as np

class LIFNeuron(BaseNeuron):
    def get_initial_states(self):
        # Vの初期値を-65.0で用意してCoreに渡す
        return {"V": np.full(self.num_neurons, -65.0)}

    def get_cuda_math(self):
        # LIFの数式（文字列）をCoreに渡す
        return """
        float dt = 0.1f;
        float tau = 20.0f;
        float dv = (-(v - (-65.0f)) + I) * (dt / tau);
        v = v + dv;
        """