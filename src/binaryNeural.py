# ************************************************************************************************************* #
#                                                                                                               #
# OBJECTIVE : DATA MINING PROJECT                                               #####      ###    ###    #      #
# AUTHORS :  VICTOR DALET , LUCA GUILLIERE, TRIBOUT NOLHAN                      #         #      #       #      #
# CREATED : 15 11 2023                                                          ####      #      #  ##   #      #
# UPDATE  : 12 11 2023                                                          #         #      #   #   #      #
#                                                                               ####    ###      #####   #.fr   #
# ************************************************************************************************************* #

import numpy as np
from tqdm import tqdm
import random


class BinaryNeural:
    def __init__(self, inputs: list[list[int]], outputs: list[int], nb_epoch: int, learning_rate: float):
        self.inputs: list[list[int]] = inputs
        self.outputs: list[int] = outputs
        self.nb_epoch: int = nb_epoch
        self.learning_rate: float = learning_rate
        self.weights: list[float] = []
        self.init_weight()

    def init_weight(self):
        for i in range(len(self.inputs[0])):
            self.weights.append(random.random())

    @staticmethod
    def sigmoid(x: float) -> int:
        return 1 / (1 + np.exp(-x))

    def predict(self, input: list[int]) -> int:
        output = 0
        for i in range(len(input)):
            output += self.weights[i] * input[i]
        return self.sigmoid(output)

    def train(self):
        for _ in tqdm(range(self.nb_epoch)):
            for i in range(len(self.inputs)):
                output = self.predict(self.inputs[i])
                error = self.outputs[i] - output
                for j in range(len(self.inputs[i])):
                    self.weights[j] += error * self.inputs[i][j] * self.learning_rate
