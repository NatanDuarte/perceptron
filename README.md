# Perceptron

perceptron training

```python
my_model = Perceptron(
        matrix=[
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ], 
        weights=[0.5, 0.7, 0.1])

my_model.train(epochs=2)

print(f'accuracy: {my_model.get_accuracy()} %')
print(f'model: {my_model.get_model()}')

# if you want to visualize your model, use this:
# my_model.plot()
```
