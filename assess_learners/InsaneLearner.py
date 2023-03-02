import numpy as np, BagLearner as bl, LinRegLearner as lrl
class InsaneLearner(object):
    def __init__(self, verbose=False):
        pass
    def author(self):
        return 'cqu41'
    def add_evidence(self, data_x, data_y):
        # create an instance of the BagLearner with 20 bags, each bag contain a linearRegressionLearner with 20 bags
        self.learners = bl.BagLearner(learner=bl.BagLearner, kwargs={'learner': lrl.LinRegLearner, 'kwargs': {}, 'bags': 20, 'boost': False, 'verbose':False}, bags=20, boost=False, verbose=False)
        self.learners.add_evidence(data_x, data_y)
        return self.learners
    def query(self, points):
        bag_outputs = []
        for i in range(20):
            bag_outputs.append(self.learners.query(points))
        return np.mean(np.array(bag_outputs), axis=0)
if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
