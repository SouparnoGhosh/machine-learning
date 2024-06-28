import pandas  # for reading csv files, data manipulation.
import numpy  # for mathematical operations
from matplotlib import pyplot as plt  # for plotting the data
from library import SinglePerceptron

# ETAs and Epochs
ETA = 0.01
EPOCHS = 100
numpy.random.seed(3)  # for reproducibility    

# #### Pre processing the data
csv_url = "breast_cancer.csv"
# reads the csv file into a pandas dataframe.
data = pandas.read_csv(csv_url)
# takes a shuffled sample of the 100% data. This shuffles the order of rows.
data = data.sample(frac=1)

print("-- Data --")
print(data)  # prints the data.
# prints the shape of the data. (rows, columns)
print(f"\nData Shape - {data.shape[0]} Rows * {data.shape[1]} Columns\n")

print("Data Description -")
print(data.describe())  # prints the summary statistics of the data.


# #### Preparing the data for training
# Desired Outputs is in the last column.
# 2 classes are there. We encode the categorical data to dummy numerical data
labels = data.iloc[:, -1].unique()
desired_outputs = numpy.where(data.iloc[:, -1] == labels[0], 1, 0)
# Gets the input features. The entire data except the last column
input_features = data.iloc[:, :-1]

split_ratio = 0.1
k = int(input_features.shape[0] * split_ratio)

train_x, train_y = input_features[:-k], desired_outputs[:-k]
test_x, test_y = input_features[-k:], desired_outputs[-k:]

print(f"Labels - {labels}")

# # #### Training our SLP
print("-- Training --")
perceptron = SinglePerceptron(train_x.shape[1], ETA)
perceptron.train(train_x.to_numpy(), train_y, EPOCHS)

# # #### Testing our SLP
print("-- Testing --")
accuracy = 0

for x, d in zip(test_x.to_numpy(), test_y):
    y = perceptron.predict(x)
    print(f"Desired Output: {d} | Actual Output: {y}")
    accuracy += 1 if d == y else 0

accuracy /= test_x.shape[0]
print(f"Accuracy for prediction: {round(accuracy * 100, 4)}%")


# # #### Plotting the accuracy
plt.figure(figsize=(20, 20))
plt.plot(range(EPOCHS), perceptron.accuracy, color='green')
plt.plot(range(EPOCHS), perceptron.error, color='red')
plt.title("Accuracy & Error w.r.t Epochs during training")
plt.xlabel("Epochs")
plt.ylabel("Accuracy, Error (in %))")
plt.grid(True, color='blue')
plt.legend(['Accuracy', 'Error'])
plt.show()

