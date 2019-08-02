import numpy as np

# Create a neural net
def create_neural_net(layer_array, input_dims):
    weights = []
    biases = []
    activations = []

    for i in range(len(layer_array)):
        node_num = layer_array[i][0]
        weights_of_layer = []
        biases_of_layer = []
        if i == 0:
            last_layer_node_number = input_dims
        else:
            last_layer_node_number = layer_array[i-1][0]

        for n in range(0,node_num):
            weights_of_node = []
            for l in range(0, last_layer_node_number):
                weights_of_node.append(1)
            weights_of_layer.append(weights_of_node)
            biases_of_layer.append(0)

        weights.append(weights_of_layer)
        biases.append(biases_of_layer)
        activations.append(layer_array[i][1])
    return [weights, biases, activations]

# Activations
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

def relu(x):
    if x < 0:
        return 0
    else:
        return x

# prediction
def predict_ratio(data, neural_net):
    weights = neural_net[0]
    biases = neural_net[1]
    activations = neural_net[2]

    layer_num = len(weights)

    for l in range(0, layer_num):
        data = np.dot(weights[l], data)
        for t in range(len(data)):
            data[t] += biases[l][t]
        if activations[l] == 'sigmoid':
            data = sigmoid(data)
        elif activations[l] == 'relu':
            data = relu(data)
        else:
            # If not identified, do it with sigmoid
            data = sigmoid(data)
            print('activation function', activations[l], 'cannot be found. Sigmoid is used')
    return data

def predict(data, neural_net):
    data = predict_ratio(data, neural_net)

    class_num = len(data)

    highest_class = None
    highest_class_probability = -1

    for i in range(0, class_num):
        if highest_class == None:
            highest_class = i
            highest_class_probability = data[i]
        elif data[i] > highest_class_probability:
            highest_class = i
            highest_class_probability = data[i]

    return highest_class, highest_class_probability

# Training
def train_network(X, Y, labels, neural_net, epochs=1000):
    for epoch in range(0, epochs):
        for d in range(0, len(X)):
            prediction = predict_ratio(X[d], neural_net)

            # Calculate total error per label
            true_prediction = []
            for i in range(0, len(labels)):
                true_prediction.append(0)
            true_prediction[labels.index(Y[d])] = 1

            errors = []
            for t in range(len(prediction)):
                errors.append(true_prediction[t] - prediction[t])
            adjust_deriv = errors * sigmoid_deriv(prediction)

            for k in range(0, len(adjust_deriv)):
                adjustment = np.dot(X[d], adjust_deriv[k])
                neural_net[0][0][k] += adjustment
    return neural_net
