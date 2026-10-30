
import pickle
from BackpropEngine import Value
import NN

n = NN.MLP(3, [4, 4, 1])

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 0.1, 0.1],
    [1.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0]

# Load saved parameters
with open("converged_params.pkl", "rb") as file:
    _parameters = pickle.load(file)

# Restore weights and biases
for layer, layer_params in zip(n.layers, _parameters):

    for neuron, neuron_params in zip(layer.neurons, layer_params):

        for w, saved_w in zip(neuron.w, neuron_params["weights"]):
            w.data = saved_w

        neuron.b.data = neuron_params["bias"]

# Forward propagation AFTER restoring parameters
ypred = [n(x) for x in xs]

# Calculate loss
loss = sum(
    (yout - ygt) ** 2
    for yout, ygt in zip(ypred, ys)
)

print("Predictions:", ypred)
print("Loss:", loss.data)
