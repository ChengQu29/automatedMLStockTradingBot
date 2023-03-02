import numpy as np

class RTLearner(object):
    def __init__(self, leaf_size, verbose=False):
        self.leaf_size = leaf_size

    def author(self):
        return "cqu41"

    def add_evidence(self, train_x, train_y):

        # if number of rows == 1, it's a leaf node
        if train_x.shape[0] == 1: return ([[None, train_y[0], np.nan, np.nan]])
        # print("data_y: ", data_y)

        if train_x.shape[0] <= self.leaf_size: return ([[None, train_y[0], np.nan, np.nan]])

        else:
            # determine best feature i to split on
            random_index = np.random.randint(0, train_x.shape[1] - 1)
            # print("feature_index: ", feature_index)
            SplitVal = np.median(train_x[:, random_index])
            # print("split value: ", SplitVal)

            left_data = train_x[train_x[:, random_index] <= SplitVal]
            right_data = train_x[train_x[:, random_index] > SplitVal]
            # cannot split any further
            if left_data.shape[0] == train_x.shape[0] or right_data.shape[0] == train_x.shape[0]:
                return np.array([[None, np.mean(train_y), np.nan, np.nan]])

            lefttree = self.add_evidence(left_data, train_y[train_x[:, random_index] <= SplitVal])
            righttree = self.add_evidence(right_data, train_y[train_x[:, random_index] > SplitVal])

            root = np.array([[random_index, SplitVal, 1, len(lefttree) + 1]])

            self.d_tree = np.vstack((root, lefttree, righttree))

            return self.d_tree

    def query(self, test_x):
        # print(self.d_tree)
        pred_y = []

        for data in test_x:
            leaf_val = self.walk_the_tree(0, data)
            pred_y.append(leaf_val)
        return np.array(pred_y)

    def walk_the_tree(self, node_index, data):

        node = self.d_tree[node_index]

        if node[0] is None:
            return node[1]

        feature_index = int(node[0])

        if data[feature_index] <= node[1]:  # value of point at feature index is less than / equal to the split value
            return self.walk_the_tree(node_index + int(node[2]), data)

        else:
            return self.walk_the_tree(node_index + int(node[3]), data)


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")