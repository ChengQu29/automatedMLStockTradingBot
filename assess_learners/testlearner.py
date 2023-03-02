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
import RTLearner as rtl
import BagLearner as bl
import InsaneLearner as il
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
    np.random.seed(244556)
    np.random.shuffle(data)
    # data = np.array(
    #     [list(map(float, s.strip().split(","))) for s in inf.readlines()]
    # )
  		  	   		  		 			  		 			     			  	 
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

    # '''
    # DTLearner
    # '''
    #
    # # create a learner and train it
    # learner = dtl.DTLearner(leaf_size=1, verbose=True)  # create a DTLearner
    # learner.add_evidence(train_x, train_y)  # train it
    # # print(learner.author())
    # # print("trained model/tree:\n",learner.d_tree)
    # # print("first node: ", learner.d_tree[0])
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

    '''
    Experiment 2
    '''
    max_leaf_size = 50
    in_sample = np.zeros(max_leaf_size + 1)
    out_of_sample = np.zeros(max_leaf_size + 1)

    for i in range(1, max_leaf_size + 1):
        learner = bl.BagLearner(learner = dtl.DTLearner, kwargs = {"leaf_size": i}, bags = 20, boost = False, verbose = False)
        learner.add_evidence(train_x, train_y)

        # evaluate in sample
        pred_y = learner.query(train_x)
        rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        in_sample[i] = rmse

        # evaluate out of sample
        pred_y = learner.query(test_x)
        rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        out_of_sample[i] = rmse

    plt.figure(2)
    plt.axis([1, max_leaf_size, 0, 0.014])

    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.title('Figure 2: RMSE vs Leaf Size - Bagging DT Learner')

    plt.plot(in_sample, label='In Sample')
    plt.plot(out_of_sample, label='Out of Sample')

    plt.legend(loc='lower right')
    plt.savefig('Figure_2.png')

    # '''
    # Insane Learner
    # '''
    #
    # learner = il.InsaneLearner(verbose=False)  # constructor
    # learner.add_evidence(train_x, train_y)  # training step
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

    '''
    Experiment 3-1
    '''
    max_leaf_size = 50
    dt_learner_time = np.zeros(max_leaf_size + 1)
    rt_learner_time = np.zeros(max_leaf_size + 1)

    for i in range(1, max_leaf_size + 1):
        learner = dtl.DTLearner(leaf_size=i)
        start_time=time.time()
        learner.add_evidence(train_x, train_y)
        end_time=time.time()

        time_diff=(end_time-start_time)*1000
        dt_learner_time[i]=time_diff

        learner = rtl.RTLearner(leaf_size=i)
        start_time = time.time()
        learner.add_evidence(train_x, train_y)
        end_time = time.time()

        time_diff = (end_time - start_time)*1000
        rt_learner_time[i] = time_diff

    plt.figure(3)
    plt.axis([1, max_leaf_size, 0, 200])

    plt.xlabel('Leaf Size')
    plt.ylabel('Time (in Milliseconds)')
    plt.title('Figure 3: Training time vs Leaf Size')

    plt.plot(dt_learner_time, label='DTLearner')
    plt.plot(rt_learner_time, label='RTLearner')

    plt.legend(loc='lower right')
    plt.savefig('Figure_3.png')

    '''
    Experiment 3-2 In sample
    '''
    max_leaf_size = 50
    dt_learner_mae_in_sample = np.zeros(max_leaf_size + 1)
    rt_learner_mae_in_sample = np.zeros(max_leaf_size + 1)

    for i in range(1, max_leaf_size + 1):
        learner = dtl.DTLearner(leaf_size=i)
        learner.add_evidence(train_x, train_y)

        pred_y = learner.query(train_x)
        mae = abs(pred_y - train_y).sum() / train_y.shape[0]
        dt_learner_mae_in_sample[i] = mae

        learner = rtl.RTLearner(leaf_size=i)
        learner.add_evidence(train_x, train_y)

        pred_y = learner.query(train_x)
        mae = abs(pred_y - train_y).sum() / train_y.shape[0]
        rt_learner_mae_in_sample[i] = mae


    plt.figure(4)
    plt.axis([1, max_leaf_size, 0, 0.015])

    plt.xlabel('Leaf Size')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Figure 4: MAE vs Leaf Size In Sample')

    plt.plot(dt_learner_mae_in_sample, label='In Sample DT')
    plt.plot(rt_learner_mae_in_sample, label='In Sample RT')

    plt.legend(loc='lower right')
    plt.savefig('Figure_4.png')

    '''
    Experiment 3-2 Out of sample
    '''
    max_leaf_size = 50
    dt_learner_mae_out_sample = np.zeros(max_leaf_size + 1)
    rt_learner_mae_out_sample = np.zeros(max_leaf_size + 1)

    for i in range(1, max_leaf_size + 1):
        learner = dtl.DTLearner(leaf_size=i)
        learner.add_evidence(train_x, train_y)

        pred_y = learner.query(test_x)
        mae = abs(pred_y - test_y).sum() / test_y.shape[0]
        dt_learner_mae_out_sample[i] = mae

        learner = rtl.RTLearner(leaf_size=i)
        learner.add_evidence(train_x, train_y)

        pred_y = learner.query(test_x)
        mae = abs(pred_y - test_y).sum() / test_y.shape[0]
        rt_learner_mae_out_sample[i] = mae

    plt.figure(5)
    plt.axis([1, max_leaf_size, 0, 0.015])

    plt.xlabel('Leaf Size')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Figure 5: MAE vs Leaf Size Out Of Sample')

    plt.plot(dt_learner_mae_out_sample, label='Out of Sample DTLearner')
    plt.plot(rt_learner_mae_out_sample, label='Out of Sample RTLearner')

    plt.legend(loc='lower right')
    plt.savefig('Figure_5.png')

    '''
    Experiment 3 R-square In Sample
    '''
    max_leaf_size = 50
    dt_learner_r_square_in_sample = np.zeros(max_leaf_size + 1)
    rt_learner_r_square_in_sample = np.zeros(max_leaf_size + 1)

    for i in range(1, max_leaf_size + 1):
        learner = dtl.DTLearner(leaf_size=i)
        learner.add_evidence(train_x, train_y)

        pred_y = learner.query(train_x)

        ss_residual = ((train_y - pred_y)**2).sum()
        ss_total = ((train_y - np.mean(train_y))**2).sum()
        r_square = 1-(ss_residual/ss_total)
        dt_learner_r_square_in_sample[i] = r_square


        learner = rtl.RTLearner(leaf_size=i)
        learner.add_evidence(train_x, train_y)

        pred_y = learner.query(train_x)

        ss_residual = ((train_y - pred_y) ** 2).sum()
        ss_total = ((train_y - np.mean(train_y)) ** 2).sum()
        r_square = 1 - (ss_residual / ss_total)
        rt_learner_r_square_in_sample[i] = r_square


    plt.figure(6)
    plt.axis([1, max_leaf_size, 0, 1])

    plt.xlabel('Leaf Size')
    plt.ylabel('Coefficient of Determination')
    plt.title('Figure 6: R-square vs Leaf Size In Sample')

    plt.plot(dt_learner_r_square_in_sample, label='In Sample DT')
    plt.plot(rt_learner_r_square_in_sample, label='In Sample RT')

    plt.legend(loc='lower right')
    plt.savefig('Figure_6.png')

    '''
        Experiment 3 R-square Out of Sample
        '''
    max_leaf_size = 50
    dt_learner_r_square_out_sample = np.zeros(max_leaf_size + 1)
    rt_learner_r_square_out_sample = np.zeros(max_leaf_size + 1)

    for i in range(1, max_leaf_size + 1):
        learner = dtl.DTLearner(leaf_size=i)
        learner.add_evidence(train_x, train_y)

        pred_y = learner.query(test_x)

        ss_residual = ((test_y - pred_y) ** 2).sum()
        ss_total = ((test_y - np.mean(test_y)) ** 2).sum()
        r_square = 1 - (ss_residual / ss_total)
        dt_learner_r_square_out_sample[i] = r_square

        learner = rtl.RTLearner(leaf_size=i)
        learner.add_evidence(train_x, train_y)

        pred_y = learner.query(test_x)

        ss_residual = ((test_y - pred_y) ** 2).sum()
        ss_total = ((test_y - np.mean(test_y)) ** 2).sum()
        r_square = 1 - (ss_residual / ss_total)
        rt_learner_r_square_out_sample[i] = r_square

    plt.figure(7)
    plt.axis([1, max_leaf_size, 0, 0.5])

    plt.xlabel('Leaf Size')
    plt.ylabel('Coefficient of Determination')
    plt.title('Figure 7: R-square vs Leaf Size Out of Sample')

    plt.plot(dt_learner_r_square_out_sample, label='Out of Sample DT')
    plt.plot(rt_learner_r_square_out_sample, label='Out of Sample RT')

    plt.legend(loc='lower right')
    plt.savefig('Figure_7.png')







