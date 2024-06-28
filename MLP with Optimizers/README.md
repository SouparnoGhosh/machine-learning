# MLP with Optimizers Project

This is the MLP (Multi-Layer Perceptron) with Optimizers project. It's an implementation of a multi-layer perceptron for classification tasks, with various optimization algorithms.

## Description

This project is implemented in Python and uses libraries such as pandas for data manipulation, numpy for mathematical operations, and matplotlib for data visualization.

The main script is `main.py`, which reads the dataset from a CSV file, preprocesses the data, trains the MLP using different optimizers, and then tests it on the test data.

## Installation

Follow these steps to install and run the project:

1. Clone the repository to your local machine.
2. Navigate to the MLP with Optimizers directory.
3. Install the required Python libraries. You can do this by running the following command in your terminal:

 ```
 pip install pandas numpy matplotlib
 ```

## Usage

To run the project, execute the `main.py` script with Python. You can do this by running the following command in your terminal:

 ```
 python main.py
 ```

This will train the MLP with different optimizers and print the accuracy of the model on the test data.

## Dataset

The project uses the `pistachio.csv` dataset, which contains data related to pistachio quality and characteristics.

## Additional Information

The project includes several optimization algorithms, including Stochastic Gradient Descent (SGD), Momentum, Root Mean Square Propagation (RMSProp), Adaptive Gradient Descent (AdaGrad), and Adaptive Momentum Estimation (Adam). These are implemented in the `optimizer.py` file.