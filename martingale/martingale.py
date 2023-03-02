""""""  		  	   		  		 			  		 			     			  	 
"""Assess a betting strategy.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
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
  		  	   		  		 			  		 			     			  	 
Student Name: Chengwen Qu  		  	   		  		 			  		 			     			  	 
GT User ID: cqu41		  	   		  		 			  		 			     			  	 
GT ID: 903756933  	   		  		 			  		 			     			  	 
"""  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
import numpy as np
import matplotlib.pyplot as plt
  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
def author():  		  	   		  		 			  		 			     			  	 
    """  		  	   		  		 			  		 			     			  	 
    :return: The GT username of the student  		  	   		  		 			  		 			     			  	 
    :rtype: str  		  	   		  		 			  		 			     			  	 
    """  		  	   		  		 			  		 			     			  	 
    return "cqu41"  # replace tb34 with your Georgia Tech username.
  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
def gtid():  		  	   		  		 			  		 			     			  	 
    """  		  	   		  		 			  		 			     			  	 
    :return: The GT ID of the student  		  	   		  		 			  		 			     			  	 
    :rtype: int  		  	   		  		 			  		 			     			  	 
    """  		  	   		  		 			  		 			     			  	 
    return 903756933 # replace with your GT ID number
  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
def get_spin_result(win_prob):  		  	   		  		 			  		 			     			  	 
    """  		  	   		  		 			  		 			     			  	 
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
    :param win_prob: The probability of winning  		  	   		  		 			  		 			     			  	 
    :type win_prob: float  		  	   		  		 			  		 			     			  	 
    :return: The result of the spin.  		  	   		  		 			  		 			     			  	 
    :rtype: bool  		  	   		  		 			  		 			     			  	 
    """  		  	   		  		 			  		 			     			  	 
    result = False
    r = np.random.random()
    # print(r)
    if r <= win_prob:
        result = True
    return result  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
def test_code():  		  	   		  		 			  		 			     			  	 
    """  		  	   		  		 			  		 			     			  	 
    Method to test your code  		  	   		  		 			  		 			     			  	 
    """
    win_prob = 9/19  # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once
    print(get_spin_result(win_prob))  # test the roulette spin

    plot_figure1(win_prob)
    plot_figure2(win_prob)
    plot_figure3(win_prob)
    #plot_figure_for_testing(win_prob)
    plot_figure4(win_prob)
    plot_figure5(win_prob)


def plot_figure5(win_prob):
    # construct a 2D array
    array = np.empty((1001, 1001))

    for index in range(1001):
        array[index] = realistic_simulator(win_prob)

    # print(array)
    arr_median = np.median(array, axis=0)
    std = np.std(array, axis=0)
    arr_median_plus_std = arr_median + std
    arr_median_minus_std = arr_median - std

    plt.axis([0, 300, -256, 100])
    plt.title("Figure 5: realistic simulator - 1000 episodes")
    plt.xlabel("Number of bets (roulette spins)")
    plt.ylabel("Total winnings")

    plt.plot(arr_median, label = "median")
    plt.plot(arr_median_plus_std, label = "median+std")
    plt.plot(arr_median_minus_std, label = "median-std")
    plt.legend()
    plt.savefig("figure5.png")
    plt.clf()


def plot_figure4(win_prob):
    # construct a 2D array
    array = np.empty((1001, 1001))

    for index in range(1001):
        array[index] = realistic_simulator(win_prob)

    # print(array)
    arr_mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    arr_mean_plus_std = arr_mean + std
    arr_mean_minus_std = arr_mean - std

    plt.axis([0, 300, -256, 100])
    plt.title("Figure 4: realistic simulator - 1000 episodes")
    plt.xlabel("Number of bets (roulette spins)")
    plt.ylabel("Total winnings")

    plt.plot(arr_mean, label = "mean")
    plt.plot(arr_mean_plus_std, label = "mean+std")
    plt.plot(arr_mean_minus_std, label = "mean-std")
    plt.legend()
    plt.savefig("figure4.png")
    plt.clf()

# def plot_figure_for_testing(win_prob):
#     plt.axis([0, 1001, -256, 100])
#     plt.title("To be deleted before submission")
#     plt.xlabel("Number of bets (roulette spins)")
#     plt.ylabel("Total winnings")
#
#     count = 0
#     for index in range(1001):
#         curr_episode = realistic_simulator(win_prob)
#         if curr_episode[-1] == 80:
#             count += 1
#         plt.plot(curr_episode)
#     print("How many times won:", count)
#     plt.show()


def realistic_simulator(win_prob):

    # add your code here to implement the experiments
    episode_winnings = 0
    array = np.zeros(1001)

    idx = 0

    while episode_winnings < 80:
        won = False
        bet_amount = 1

        while not won:
            # todo: wager bet amount on black
            if idx >= 1001:
                return array
            else:
                array[idx] = episode_winnings
                won = get_spin_result(win_prob)
                idx += 1
                #print(idx)

                if won == True:
                    episode_winnings = episode_winnings + bet_amount
                else:
                    episode_winnings = episode_winnings - bet_amount
                    bet_amount *= 2

                    if episode_winnings <= -256:
                        array[idx:] = episode_winnings
                        return array
                    if episode_winnings + 256 < bet_amount:
                        bet_amount = 256 + episode_winnings


    array[idx:] = episode_winnings

    return array

def plot_figure3(win_prob):


    # construct a 2D array
    array = np.empty((1001, 1001))

    for index in range(1001):
        array[index] = simulator(win_prob)

    # print(array)
    arr_median = np.median(array, axis=0)
    std = np.std(array, axis=0)
    arr_median_plus_std = arr_median + std
    arr_median_minus_std = arr_median - std

    plt.axis([0, 300, -256, 100])
    plt.title("Figure 3: simple simulator - 1000 episodes")
    plt.xlabel("Number of bets (roulette spins)")
    plt.ylabel("Total winnings")

    plt.plot(arr_median, label = "median")
    plt.plot(arr_median_plus_std, label = "median+std")
    plt.plot(arr_median_minus_std, label = "median-std")
    plt.legend()
    plt.savefig("figure3.png")
    plt.clf()

def plot_figure2(win_prob):
    plt.axis([0, 300, -256, 100])
    plt.title("Figure 2: simple simulator - 1000 episodes")
    plt.xlabel("Number of bets (roulette spins)")
    plt.ylabel("Total winnings")

    # construct a 2D array
    array = np.empty((1001, 1001))

    for index in range(1001):
        array[index] = simulator(win_prob)

    #print(array)
    arr_mean = np.empty(1001)
    arr_mean_plus_std = np.empty(1001)
    arr_mean_minus_std = np.empty(1001)

    for i in range(1001):
        arr_mean[i] = np.mean(array[:,i])
        arr_mean_plus_std[i] = arr_mean[i] + np.std(array[:,i])
        arr_mean_minus_std[i] = arr_mean[i] - np.std(array[:, i])

    plt.plot(arr_mean, label = "mean")
    plt.plot(arr_mean_plus_std, label = "mean+std")
    plt.plot(arr_mean_minus_std, label= "mean-std")
    plt.legend()
    plt.savefig("figure2.png")
    plt.clf()

def plot_figure1(win_prob):
    plt.axis([0, 300, -256, 100])
    plt.title("Figure 1: simple simulator - 10 episodes")
    plt.xlabel("Number of bets (roulette spins)")
    plt.ylabel("Total winnings")

    #count = 0
    for index in range(10):
        curr_episode = simulator(win_prob)
        #if curr_episode[-1] == 80:
        #    count += 1
        plt.plot(curr_episode)

    #print("how many times won:", count)
    plt.savefig("figure1.png")
    plt.clf()

def simulator(win_prob):

    # add your code here to implement the experiments
    episode_winnings = 0

    array = np.zeros(1001)

    array[0] = episode_winnings

    idx = 0
    while episode_winnings < 80:
        won = False
        bet_amount = 1

        while not won:
            # todo: wager bet amount on black

            array[idx] = episode_winnings
            won = get_spin_result(win_prob)
            idx += 1
            if won == True:
                episode_winnings = episode_winnings + bet_amount
            else:
                episode_winnings = episode_winnings - bet_amount
                bet_amount *= 2

    array[idx:] = episode_winnings
    return array

  		  	   		  		 			  		 			     			  	 
if __name__ == "__main__":  		  	   		  		 			  		 			     			  	 
    test_code()  		  	   		  		 			  		 			     			  	 
