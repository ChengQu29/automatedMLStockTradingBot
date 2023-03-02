"""
Test a learner.  (c) 2015 Tucker Balch  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		  		 			  		 			     			  	 
All Rights Reserved  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  		 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		  		 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  		 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 			  		 			     			  	 
or edited.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  		 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		  		 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 			  		 			     			  	 
GT honor code violation.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		  		 			  		 			     			  	 
"""  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
import math  		  	   		  		 			  		 			     			  	 
import sys  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
import numpy as np  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
import LinRegLearner as lrl
import DTLearner as dtl

import matplotlib.pyplot as plt
import time
  		  	   		  		 			  		 			     			  	 
if __name__ == "__main__":  		  	   		  		 			  		 			     			  	 
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    inf = open(sys.argv[1])
    #inf = open("Data/simple2.csv")
    data = np.genfromtxt(inf, delimiter=',')
    # strip first row and column
    data = data[1:, :]
    data = data[:, 1:]
    #print(data)

    #shuffle data
    np.random.seed(903756933)
    # np.random.shuffle(data)

  		  	   		  		 			  		 			     			  	 
    # compute how much of the data is training and testing  		  	   		  		 			  		 			     			  	 
    train_rows = int(0.6 * data.shape[0])  		  	   		  		 			  		 			     			  	 
    test_rows = data.shape[0] - train_rows  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
    # separate out training and testing data  		  	   		  		 			  		 			     			  	 
    train_x = data[:train_rows, 0:-1]
    #print("train_x: ", train_x)
    train_y = data[:train_rows, -1]
    #print("train_y: ", train_y)
    test_x = data[train_rows:, 0:-1]  		  	   		  		 			  		 			     			  	 
    test_y = data[train_rows:, -1]  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
    #print(f"test_x shape: {test_x.shape}")
    #print(f"test_y shape: {test_y.shape}")
  		  	   		  		 			  		 			     			  	 
    # # create a learner and train it
    # learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    # learner.add_evidence(train_x, train_y)  # train it
    # print(learner.author())
  	#
    # # evaluate in sample
    # pred_y = learner.query(train_x)  # get the predictions
    # rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    # print()
    # print("In sample results")
    # print(f"RMSE: {rmse}")
    # c = np.corrcoef(pred_y, y=train_y)
    # print(f"corr: {c[0,1]}")
  	#
    # # evaluate out of sample
    # pred_y = learner.query(test_x)  # get the predictions
    # rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    # print()
    # print("Out of sample results")
    # print(f"RMSE: {rmse}")
    # c = np.corrcoef(pred_y, y=test_y)
    # print(f"corr: {c[0,1]}")

    '''
    DTLearner
    '''

    # create a learner and train it
    learner = dtl.DTLearner(leaf_size=1, verbose=True)  # create a DTLearner
    learner.add_evidence(train_x, train_y)  # train it
    # print(learner.author())
    # print("trained model/tree:\n",learner.d_tree)
    # print("first node: ", learner.d_tree[0])

    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0, 1]}")

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0, 1]}")

    '''
    Experiment1
    '''
    max_leaf_size = 50
    in_sample = np.zeros(max_leaf_size + 1)
    out_of_sample = np.zeros(max_leaf_size + 1)

    for i in range(1, max_leaf_size + 1):
        learner = dtl.DTLearner(leaf_size=i)
        learner.add_evidence(train_x, train_y)

        # evaluate in sample
        pred_y = learner.query(train_x)
        rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        in_sample[i] = rmse

        # evaluate out of sample
        pred_y = learner.query(test_x)
        rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        out_of_sample[i] = rmse

    plt.figure(1)
    plt.axis([1, max_leaf_size, 0, 0.014])

    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.title('Figure 1: RMSE vs Leaf Size - DT Learner')

    plt.plot(in_sample, label='In Sample')
    plt.plot(out_of_sample, label='Out of Sample')

    plt.legend(loc='lower right')
    plt.savefig('Figure_1.png')

    # '''
    # bagging
    # '''
    # # create a learner and train it
    # learner = bl.BagLearner(learner = dtl.DTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False)
    # learner.add_evidence(train_x, train_y)  # train it
    # print(learner.author())
    #
    # # evaluate in sample
    # pred_y = learner.query(train_x)  # get the predictions
    # rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    # print()
    # print("In sample results")
    # print(f"RMSE: {rmse}")
    # c = np.corrcoef(pred_y, y=train_y)
    # print(f"corr: {c[0, 1]}")
    #
    # # evaluate out of sample
    # pred_y = learner.query(test_x)  # get the predictions
    # rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    # print()
    # print("Out of sample results")
    # print(f"RMSE: {rmse}")
    # c = np.corrcoef(pred_y, y=test_y)
    # print(f"corr: {c[0, 1]}")








