from perceptron import Perceptron

def main():
    example_func()

def example_func():

    # define your dataset
    matrix = [
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 1]
    ]

    # and your weights
    weights = [0.5, 0.7, 0.1]

    # then, instantiate your model
    my_model = Perceptron(matrix=matrix, weights=weights)

    # and train it
    my_model.train(epochs=2)

    print(f'accuracy: {my_model.get_accuracy()}%')
    print(f'model: {my_model.get_model()}')

    # if you want to visualize your model, use this:
    my_model.plot()

main()
