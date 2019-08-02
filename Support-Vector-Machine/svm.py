import numpy as np
import matplotlib.pyplot as plt

def weight_graph(w0array, w1array, number_of_weights_to_graph=100):
    # You cannot see anything in the graph of 10000 numbers :)
    epochs = len(w0array)

    # It will divide epochs to this number
    num_per_epoch = epochs/number_of_weights_to_graph

    w0_to_graph = []
    w1_to_graph = []
    epoch_to_graph = []

    for i in range(number_of_weights_to_graph):
        epoch_to_graph.append(int(num_per_epoch*i))
        w0_to_graph.append(w0array[int(num_per_epoch*i)])
        w1_to_graph.append(w1array[int(num_per_epoch*i)])

    plt.plot(epoch_to_graph, w0_to_graph, 'r',epoch_to_graph, w1_to_graph,'b')


def train_svm(X, Y, epochs=10000, learning_rate=1):
    #Initialize our SVMs weight vector with zeros (3 values)
    w = np.zeros(len(X[0]))

    # See the change
    w0_per_epoch = []
    w1_per_epoch = []

    # Training
    print("starts training")
    for epoch in range(1, epochs):
        for i, x in enumerate(X):
            # It there is an error
            if (Y[i] * np.dot(X[i], w)) < 1:
                w = w + learning_rate * ((X[i] * Y[i]) + (-2 * (1/epochs) * w))
            else:
                w = w + learning_rate * (-2 * (1/epochs) * w)
        w0_per_epoch.append(w[0])
        w1_per_epoch.append(w[1])

    weight_graph(w0array, w1array)
    return w

def predict(X, w):
    Y = np.dot(X, w))
    return Y

def show_svm_graph(X, Y, w):
    # For every point mark point with - if label is -1 and mark with + if label is 1
    for i in range(len(X)):
        if Y[i] == -1:
            plt.scatter(X[i][0], X[i][1], s=120, marker='_', linewidths=2)
        else:
            plt.scatter(X[i][0], X[i][1], s=120, marker='+', linewidths=2)

    # Print the hyperplane calculated by svm_sgd()
    x2=[w[0]*0.65,w[1],-w[1],w[0]]
    x3=[w[0]*0.65,w[1],w[1],-w[0]]

    x2x3 =np.array([x2,x3])
    X,Y,U,V = zip(*x2x3)
    ax = plt.gca()
    ax.quiver(X,Y,U,V,scale=1, color='blue')
