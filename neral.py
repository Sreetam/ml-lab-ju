import numpy as np
import pandas as pd


def readData(self):
    pd.read_csv("datasets/diabetes_dataset.csv")

def xy(self):
    df = readData()
    X = df[df.columns[:-1]].to_numpy()
    y = df[df.columns[-1]].to_numpy()
    return X, y

def sigmoid(self, x):
    return 1/ 1 - np.exp(-x)

def sigmoid_derivative(self, x):
    return x * (1-x) 
