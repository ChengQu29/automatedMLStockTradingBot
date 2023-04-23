import math
import sys

import numpy as np

import LinRegLearner as lrl
import DTLearner as dtl
import RTLearner as rtl
import BagLearner as bl
import InsaneLearner as il
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":


    inf = open("Data/simple2.csv")
    data = np.genfromtxt(inf, delimiter=',')
    # strip first row and column

    print(data)

    # data = np.array(
    #     [list(map(float, s.strip().split(","))) for s in inf.readlines()]
    # )

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    train_x = data[:train_rows, 0:-1]
    print("train_x: ", train_x)
    train_y = data[:train_rows, -1]
    print("train_y: ", train_y)
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    print(f"test_x shape: {test_x.shape}")
    print(f"test_y shape: {test_y.shape}")

    # create a learner and train it
    learner = dtl.DTLearner(leaf_size=1)  # create a learner
    learner.add_evidence(train_x, train_y)  # train it
    print(learner.d_tree)

