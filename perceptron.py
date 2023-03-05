# import matplotlib.pyplot as plt
# import numpy as np

class Perceptron():
    def __init__(self, matrix:list, weights:list,
                bias:float=-1, learning_rate:float=0.5) -> None:
        self.matrix = matrix
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate

    def train(self, epochs:int) -> None:
        """# Trains your model to (try) fitting your problem"""
        for _ in range(epochs):
            for sample in self.matrix:
                expected, prediction, weighted_sum, inputs = self._predict(sample)
                if prediction != weighted_sum:
                    self._adjust_weights(self.weights, self.learning_rate, 
                                        inputs, expected, prediction)

    def _step_activation_function(self, weighted_sum:float, threshold:float=0.0):
        return 1 if weighted_sum > threshold else 0

    def _adjust_weights(self, weights:list, learning_rate:float, 
                        inputs:list, expected:int, prediction:int):
        for i in range(len(weights)):
            weights[i] = weights[i] + learning_rate * (expected - prediction) * inputs[i]
            weights[i] = float(f'{weights[i]:.1f}')

    def _get_weighted_sum(self, weights, inputs):
        weighted_sum = 0
        for x, w in zip(inputs, weights):
            weighted_sum += x * w
        return weighted_sum

    def get_accuracy(self):
        correct_answers = 0
        for sample in self.matrix:
            expected, prediction, _, _ = self._predict(sample)
            if prediction == expected:
                correct_answers += 1
        return float(f'{(correct_answers / float(len(self.matrix)) * 100):.2f}')

    def predict(self, x1:float, x2:float):
        return self._step_activation_function(
            self.weights[0] * x1 + self.weights[1] * x2 + self.weights[2] * self.bias
        )

    def _predict(self, sample):
        inputs = sample[:-1]
        inputs.append(self.bias)
        expected = sample[-1]
        weighted_sum = self._get_weighted_sum(self.weights, inputs)
        prediction = self._step_activation_function(weighted_sum)
        return expected, prediction, weighted_sum, inputs

    def __str__(self) -> str:
        output = {
            'weight_x': self.weights[0],
            'weight_y': self.weights[1],
            'weight_bias': self.weights[2],
            'bias': self.bias,
            'learning_rate': self.learning_rate,
        }
        return f'{output}'

    def get_model(self):
        return f'Y = g(x1 * {self.weights[0]} + x2 * {self.weights[1]} + {self.weights[2]} * {self.bias})'

    # def plot(self):
    #     X = np.array(self.matrix)
    #     Y = X[:, -1]
    #     X = X[:, :-1]
    #     plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, edgecolor='k')

    #     xmin, xmax = -10, 10
    #     ymin, ymax = -10, 10

    #     x = np.linspace(-5, 5, 50)
    #     y = (-self.weights[0]*x -self.bias*self.weights[2]) / self.weights[1]

    #     plt.axvline(0, -10, 10, color='k', linewidth=1)
    #     plt.axhline(0, -10, 10, color='k', linewidth=1)
    #     plt.plot(x, y, label='_nolegend_')

    #     plt.xlim(xmin, xmax)
    #     plt.ylim(ymin, ymax)
    #     plt.show()
