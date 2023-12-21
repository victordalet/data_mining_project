import numpy as np
from tqdm import tqdm
import random


class BinaryNeural:
    def __init__(self, inputs: list[list[int]],
                 outputs: list[int],
                 nb_epoch: int,
                 learning_rate: float):
        self.inputs: list[list[int]] = inputs
        self.outputs: list[int] = outputs
        self.nb_epoch: int = nb_epoch
        self.learning_rate: float = learning_rate
        self.weights: list[float] = []
        self.path: str = 'weights.json'

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


    def train(self) -> None:
        if not self.check_json_file_is_empty():
            print(self.get_score())
            return self.get_weights_in_json_file()
          
        for _ in tqdm(range(self.nb_epoch)):
            for i in range(len(self.inputs)):
                output = self.predict(self.inputs[i])
                error = self.outputs[i] - output
                for j in range(len(self.inputs[i])):
                    self.weights[j] += (error * self.inputs[i][j]
                                        * self.learning_rate)
        print(self.get_score())
        self.save_weights_to_json_file()

    def get_score(self) -> float:
        score = 0
        for i in tqdm(range(len(self.inputs))):
            score += (self.predict(self.inputs[i]) - self.outputs[i]) ** 2
        return score / len(self.inputs)

    def save_weights_to_json_file(self) -> None:
        with open(self.path, 'w') as f:
            f.write(str(self.weights))

    def get_weights_in_json_file(self) -> None:
        with open(self.path, 'r') as f:
            self.weights = eval(f.read())

    def check_json_file_is_empty(self) -> bool:
        with open(self.path, 'r') as f:
            return f.read() == ''
