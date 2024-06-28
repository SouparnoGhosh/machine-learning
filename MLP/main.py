import numpy as np
import pandas
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from MLP import MLP
from helpers import normalize_data, big_print, max_print

if __name__ == "__main__":
    # Read the csv file
    url = "pistachio.csv"
    data = pandas.read_csv(url)

    # Shuffle the data
    np.random.seed(2192)
    data = data.sample(frac=1)


    # Feature columns - all columns except last one
    X = data.iloc[:, :-1]

    # Target column - last column
    Y = data.iloc[:, -1]
    Y = Y.replace({'Kirmizi_Pistachio':0, 'Siit_Pistachio':1})
    
    # No of input neurons = no of feature columns in dataset
    m = X.shape[1]
    # No of output neurons = 1 as there are only 2 classes 
    n = 1
    # Epochs
    EPOCHS = 300

    # Normalization
    X_norm = normalize_data(X)

    split_ratio = 0.2
    random_state = 1

    # #* Training and testing data
    # separating the data for training and testing
    # random state is for reproducibility 
    train_x, test_x, train_y, test_y = train_test_split(X_norm, Y, test_size=split_ratio, random_state=random_state)

    model = MLP(m, [11,11,11,11], n)

    accuracy_list, error_list = model.train(train_x.values, train_y.values, test_x.values, test_y.values, EPOCHS);


    #* Plotting the data
    plt.figure(figsize=(20,10))

    # plt.subplot(1,2,1)
    plt.plot(accuracy_list)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Accuracy vs Epochs')
    plt.grid(True)

    plt.tight_layout(pad=4.0)
    plt.show()

    max_print(accuracy_list)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # # plt.subplot(1,2,2)
    # plt.plot(error_list)
    # plt.xlabel('Epochs', fontsize=14)
    # plt.ylabel('Error', fontsize=14)
    # plt.title('Error vs Epochs')
    # plt.grid(True)

    