import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn import tree
import seaborn as sns

class Node:
    def __init__(self, label):
        self.label = label # nhãn của node nếu node đó là node lá
        self.best_attribute_idx = 0 # thuộc tính để phân tách dữ liệu thành 2 node con
        self.best_splitting_threshold = 0 # ngưỡng phân tách dữ liệu thành 2 node con
        self.left_branch = None # node con trái
        self.right_branch = None # node con phải
        self.gini = 0 # giá trị gini

    def set_left_branch(self, left_branch):
        self.left_branch = left_branch
    
    def set_right_branch(self, right_branch):
        self.right_branch = right_branch

def calc_gini(freq_list):
    arr = np.array(freq_list)

    no_samples = np.sum(arr)
    
    tmp = 0
    for freq in arr:
        tmp += (freq/no_samples) ** 2
    return 1 - tmp



class CartDecisionTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
    
    def _build_tree(self, X, y, depth=0):
        no_samples_by_class = [np.sum(y == i) for i in range(self.no_classes)]
        label_mode = np.argmax(no_samples_by_class)
        root = Node(label_mode)
        
        if depth > self.max_depth:
            return root
        
        best_attribute_idx, best_spliting_threshold, min_gini = self._split(X, y)

        if best_attribute_idx is None:
            return root
        
        pivot_idx = X[:, best_attribute_idx] < best_spliting_threshold

        root.best_attribute_idx = best_attribute_idx
        root.best_splitting_threshold = best_spliting_threshold
        root.gini = min_gini
        
        X_left_data, y_left_target = X[pivot_idx], y[pivot_idx]
        X_right_data, y_right_target = X[~pivot_idx], y[~pivot_idx]

        left_branch = self._build_tree(X_left_data, y_left_target, depth + 1)
        right_branch = self._build_tree(X_right_data, y_right_target, depth + 1)

        root.set_left_branch(left_branch)
        root.set_right_branch(right_branch)

        return root

    def _calc_average_gini(self, left_gini, no_samples_in_left_parititon, right_gini, no_samples_in_right_parititon):
        no_samples = no_samples_in_left_parititon + no_samples_in_right_parititon
        res = (no_samples_in_left_parititon / no_samples) * left_gini + \
              (no_samples_in_right_parititon / no_samples) * right_gini
        return res

    def _calc_threshold_by_type(self, left_value, right_value, type="mean"):
        res = left_value + (right_value - left_value) / 2
        return res

    def _split(self, X, y):
        no_samples = len(y)
        if no_samples <= 1:
            return None, None, 0

        no_samples_by_class = [np.sum(y == i) for i in range(self.no_classes)]  
        min_gini = calc_gini(no_samples_by_class)
        best_attribute_idx = None
        best_splitting_threshold = None

        for attr_idx in range(self.no_features):
            value_list, class_list = zip(*sorted(zip(X[:, attr_idx], y)))

            left_parititon = [0 for i in range(self.no_classes)]
            right_partition = no_samples_by_class.copy()

            for row_idx in range(1, no_samples):
                label = y[row_idx - 1]
                
                left_parititon[label] += 1
                right_partition[label] -= 1

                no_samples_in_left_parititon = row_idx
                no_samples_in_right_parititon = no_samples - no_samples_in_left_parititon

                left_gini = calc_gini(left_parititon)
                right_gini = calc_gini(right_partition)

                average_gini = \
                    self._calc_average_gini(left_gini, no_samples_in_left_parititon, \
                        right_gini, no_samples_in_right_parititon)
                
                curr_value = value_list[row_idx - 1]
                next_value = value_list[row_idx]

                if curr_value != next_value:
                    if min_gini > average_gini:
                        min_gini = average_gini
                        best_attribute_idx = attr_idx
                        best_splitting_threshold = self._calc_threshold_by_type(curr_value, next_value, type="mean")

        return best_attribute_idx, best_splitting_threshold, min_gini  

    def fit(self, X, y):
        self.no_classes = len(set(y))
        self.no_features = X.shape[1]
        self.root = self._build_tree(X, y)

    def _predict(self, sample):
        curr_node = self.root
        while curr_node.left_branch:
            if sample[curr_node.best_attribute_idx] < curr_node.best_splitting_threshold:
                curr_node = curr_node.left_branch
            else:
                curr_node = curr_node.right_branch
        return curr_node.label
    
    def predict(self, test_data):
        y_pred = []

        for sample in test_data:
            y_pred.append(self._predict(sample))
        return y_pred

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    print("Kích thước bộ dữ liêu:", X.shape)
    skf = StratifiedKFold(n_splits=5)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        # clf = CartDecisionTree(max_depth = 20)
        clf = tree.DecisionTreeClassifier(max_depth=20)
        train_data = X[train_idx]
        train_target = y[train_idx]
        test_data = X[test_idx]
        test_target = y[test_idx]

        clf.fit(train_data, train_target)
        y_pred = clf.predict(test_data)
        score = accuracy_score(y_pred, test_target)
        scores.append(score)
    scores = np.array(scores)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))