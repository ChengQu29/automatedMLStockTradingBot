import numpy as np
class BagLearner(object):
    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):

        learners = []
        # put a learner in each bag
        for i in range(0, bags):
            learners.append(learner(**kwargs))

        self.learners = learners
    def author(self):
        return "cqu41"
    def add_evidence(self, data_x, data_y):

        data = np.column_stack((data_x, data_y))

        n = data.shape[0]

        for learner in self.learners:
            bag = np.zeros(shape=(0, data.shape[1])) # 0 rows and

            for i in range(n):

                index = np.random.randint(0, n)

                bag = np.append(bag, [data[index]], axis=0)

            bag_x = bag[:, 0:-1]
            bag_y = bag[:, -1]

            learner.add_evidence(bag_x, bag_y)
    def query(self, points):
        result = []

        for learner in self.learners:
            pred_y = learner.query(points)
            result.append(pred_y)

        return np.mean(np.array(result), axis=0)

if __name__ == '__main__':
    print("the secret clue is 'zzyzx'")