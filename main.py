import matplotlib.pyplot as plt

from opt_utils import model
from opt_utils import load_dataset, predict, plot_decision_boundary, predict_dec


if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    # Model with All Optimizations 
    train_X, train_Y = load_dataset()

    # Mini Batch Gradient Descent Example
    # trains 3-layer model
    print("MODEL WITH MINI BATCH GRADIENT DESCENT")
    layers_dims = [train_X.shape[0], 5, 2, 1]
    parameters = model(train_X, train_Y, layers_dims, optimizer = "gd")

    # Predicts
    predictions = predict(train_X, train_Y, parameters)

    # Plots decision boundary
    plt.title("Model with Mini Batch Gradient Descent Optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5,2.5])
    axes.set_ylim([-1,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


    print("MODEL WITH MINI BATCH MOMENTUM")
    # Mini Batch Gradient Descent with Momentum Example
    # trains 3-layer model
    layers_dims = [train_X.shape[0], 5, 2, 1]
    parameters = model(train_X, train_Y, layers_dims, beta = 0.9, optimizer = "momentum")

    # Predicts
    predictions = predict(train_X, train_Y, parameters)

    # Plots decision boundary
    plt.title("Model with Mini Batch Momentum optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5,2.5])
    axes.set_ylim([-1,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


    print("MODEL WITH MINI BATCH ADAM")
    # Mini Batch Gradient Descent with Adam Example
    # trains 3-layer model
    layers_dims = [train_X.shape[0], 5, 2, 1]
    parameters = model(train_X, train_Y, layers_dims, optimizer = "adam")

    # Predicts
    predictions = predict(train_X, train_Y, parameters)

    # Plots decision boundary
    plt.title("Model with Mini Batch Adam optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5,2.5])
    axes.set_ylim([-1,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)