import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from optimizer import SGD, RMSProp, Adam, Momentum, AdaGrad
from activation import Sigmoid
from loss import MSE_Loss
from utils import normalize_data, print_tables
from MLP import MLP


if __name__ == "__main__":

    np.random.random(24)
    random_state = np.random.RandomState(1)

    url = "pistachio.csv"

    data = pandas.read_csv(url)

    EPOCHS = 500

    m = data.columns[:-1].shape[0]  # Number of features
    n = 1  # Number of outputs

    # Feature columns - all columns except last one
    X = data.iloc[:, :-1]
    X_norm = normalize_data(X, data.iloc[:, :-1])

    # Target column - last column
    Y = data.iloc[:, -1]
    Y = Y.replace({'Kirmizi_Pistachio': 0, 'Siit_Pistachio': 1})
    Y_encoded = Y.values.reshape(-1, 1)

    split_ratio = 0.3

    train_x, test_x, train_y, test_y = train_test_split(
        X_norm, Y_encoded, test_size=split_ratio, random_state=random_state
    )

    # Instances of different optimizers
    optimizers = {
        "Gradient Descent": SGD(lr=0.01),
        "Momentum": Momentum(lr=0.1, alpha=0.5),
        "RMSProp": RMSProp(lr=0.01, epsilon=0.1, alpha=0.95),
        "AdaGrad": AdaGrad(lr=0.1, epsilon=0.1),
        "Adam": Adam(lr=0.1, epsilon=0.1, alpha=0.95, beta=0.5),
    }

    # Store metrics for each optimizer
    optimizer_metrics = {}

    for optimizer_name, optimizer in optimizers.items():
        model = MLP(m, [11], n, MSE_Loss, optimizer, Sigmoid)
        optimizer_metrics[optimizer_name] = model.train(
            train_x.values, train_y, test_x.values, test_y, EPOCHS, batch_size = 8
        )
        
    print_tables(optimizers, optimizer_metrics)

    fig = plt.figure(figsize=(20, 10))

    # Create the first subplot for loss
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
    for optimizer_name in optimizers.keys():
        plt.plot(
            optimizer_metrics[optimizer_name]["loss"], label=optimizer_name)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss v/s Epochs")
    plt.grid(True)  # Add grid



    # Create the second subplot for accuracy
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
    for optimizer_name in optimizers.keys():
        plt.plot(optimizer_metrics[optimizer_name]
                 ["accuracy"], label=optimizer_name)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy v/s Epochs")
    plt.grid(True)  # Add grid

    # Show the plots
    plt.tight_layout()
    plt.show()
