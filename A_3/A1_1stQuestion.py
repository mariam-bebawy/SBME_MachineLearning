import pandas as pd
import numpy as np
import time
# import sklearn
from sklearn import datasets, tree
from sklearn.model_selection import train_test_split


class node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.leftChild = left
        self.rightChild = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class decisionTree:
    def __init__(self, maxDepth=100, minSplit=2):
        self.maxDepth = maxDepth
        self.minSplit = minSplit
        self.root = None

    def _finished(self, depth):
        if (depth >= self.maxDepth or self.nLabels == 1
                or self.nSamples < self.minSplit):
            return True
        return False

    def fit(self, x, y):
        self.root = self._buildTree(x, y)

    def _entropy(self, y):
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy

    def _split(self, x, threshold):
        leftIndex = np.argwhere(x <= threshold).flatten()
        rightIndex = np.argwhere(x > threshold).flatten()
        return leftIndex, rightIndex

    def _IG(self, x, y, threshold):
        parentLoss = self._entropy(y)
        leftIndex, rightIndex = self._split(x, threshold)
        n, nLeft, nRight = len(y), len(leftIndex), len(rightIndex)

        if nLeft == 0 or nRight == 0:
            return 0

        childLoss = (nLeft / n) * self._entropy(
            y[leftIndex]) + (nRight / n) * self._entropy(y[rightIndex])
        return parentLoss - childLoss

    def _bestSplit(self, x, y, features):
        split = {'score': -1, 'feat': None, 'thresh': None}

        for feat in features:
            X_feat = x[:, feat]
            thresholds = np.unique(X_feat)
            for threshold in thresholds:
                score = self._IG(X_feat, y, threshold)

                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['thresh'] = threshold

        return split['feat'], split['thresh']

    def _buildTree(self, x, y, depth=0):
        self.nSamples, self.nFeatures = x.shape
        self.nLabels = len(np.unique(y))

        # get best split
        rndFeatures = np.random.choice(self.nFeatures,
                                     self.nFeatures,
                                     replace=False)
        bestFeature, bestThreshold = self._bestSplit(x, y, rndFeatures)

        # stopping criteria
        if self._finished(depth):
            # maxCountLabel ----> most common label
            maxCountLabel = np.argmax(np.bincount(y))
            return node(value=maxCountLabel)

        leftIndex, rightIndex = self._split(x[:, bestFeature], bestThreshold)
        leftChild = self._buildTree(x[leftIndex, :], y[leftIndex], depth + 1)
        rightChild = self._buildTree(x[rightIndex, :], y[rightIndex],
                                       depth + 1)
        return node(bestFeature, bestThreshold, leftChild, rightChild)

    def _traverseTree(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverseTree(x, node.leftChild)
        return self._traverseTree(x, node.rightChild)

    def predict(self, x):
        predictions = [self._traverseTree(element, self.root) for element in x]
        return np.array(predictions)


def main():
    df = pd.read_csv('cardio_train.csv', sep=';')
    df = df.drop(columns=['id','age','height','weight','gender','smoke'])
    print(df[0:0])
    dValues = np.array(df[df.columns[:-1]])
    dLabels = np.array(df.cardio)

    X_train, X_test, y_train, y_test = train_test_split(dValues, dLabels, random_state=0)

    # IMPLEMENTED DECISION TREE
    dTree = decisionTree(maxDepth=5)
    # SKLEARN DECISION TREE
    skTree = tree.DecisionTreeClassifier(criterion="entropy")
    
    # TRAINING
    start = time.time()
    dTree.fit(X_train, y_train)
    end = time.time()
    print("training time of implemented model: ", round((end - start),4), "s")
    start = time.time()
    skTree.fit(X_train, y_train)
    end = time.time()
    print("training time of sklearn model: ", round((end - start),4), "s")

    # PREDICTION
    start = time.time()
    y_pred1 = dTree.predict(X_test)
    end = time.time()
    print("prediction time of implemented model: ", round((end - start),4)*1000, "ms")
    start = time.time()
    y_pred2 = skTree.predict(X_test)
    end = time.time()
    print("prediction time of sklearn model: ", round((end - start),4)*1000, "ms")

    # ACCURACY
    accuracy = np.sum(y_test == y_pred1)/len(y_test)
    print("accuracy of implemented model: ", round(accuracy,3)*100, "%")
    accuracy = np.sum(y_test == y_pred2)/len(y_test)
    print("accuracy of sklearn model: ", round(accuracy,3)*100, "%")


if __name__ == '__main__':
    main()