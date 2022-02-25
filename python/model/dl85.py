import numpy as np
import pandas as pd
import time
from math import ceil, floor, log
from sklearn.metrics import confusion_matrix, accuracy_score
from dl85 import DL85Classifier
from model.tree_classifier import TreeClassifier
from model.encoder import Encoder


class DL85:
    def __init__(self, preprocessor="none", regularization=None, depth=2**30, support=1, time_limit=900, warm = None):
        # Precompute serialized configuration
        self.preprocessor = preprocessor
        self.regularization = regularization
        self.depth = depth if depth != 0 else 2**30 #use depth 0 to convey no depth limit
        self.support = support
        self.time_limit = time_limit
        self.warm = warm #represents the reference model's performance on the training set points
        self.lb = -1
        self.ub = -1
        self.loss = -1
        self.reported_loss = -1

    def fit(self, X, y):
        self.shape = X.shape
        (n, m) = self.shape

        encoder = Encoder(X.values[:,:], header=X.columns[:], mode=self.preprocessor, target=y[y.columns[0]])
        headers = encoder.headers

        X = pd.DataFrame(encoder.encode(X.values[:,:]), columns=encoder.headers)
        y = y.reset_index(drop=True)
        self.encoder = encoder

        if not self.regularization is None:
            #in gosdt, regularization automatically implies a bound on depth - we apply that here too for fairness
            # depth_constraint_from_reg = ceil(1 / self.regularization) if self.regularization > 0 else 2**30
            # depth = min(depth_constraint_from_reg, self.depth) #if the user also specified a depth constraint, we should use the tighter of these two constraints
            support = ceil(self.regularization * n)

            # def error(sup_iter):
            #     supports = list(sup_iter)
            #    maxindex = np.argmax(supports)
            #    return sum(supports) - supports[maxindex] + self.regularization * n, maxindex

            clf = DL85Classifier(
                # fast_error_function=error,
                iterative=False,
                time_limit=self.time_limit,
                min_sup=support,
                max_depth=self.depth-1 #dl8.5's notion of depth excludes the root node, so we correct for an OBOE here
            )
        else:
            clf = DL85Classifier(
                iterative=False,
                time_limit=self.time_limit,
                min_sup=self.support,
                max_depth=self.depth-1 #as above, correct for OBOE
            )

        start = time.perf_counter()
        clf.fit(X, y.values.ravel(), warm=self.warm)
        self.time = time.perf_counter() - start
        self.utime = self.time
        self.stime = 0
        self.space = clf.lattice_size_
        source = self.__translate__(clf.tree_)
        self.tree = TreeClassifier(source, encoder=encoder)
        self.tree.__initialize_training_loss__(X, y)
        return self

    def __translate__(self, node):
        (n, m) = self.shape

        if "class" in node:
            return {
                "name": "class",
                "prediction": node["class"],
                "loss": node["error"] / n,
                "complexity": self.regularization
            }
        elif "value" in node:
            return {
                "name": "class",
                "prediction": node["value"],
                "loss": node["error"] / n,
                "complexity": self.regularization
            }
        elif "feat" in node:
            return {
                "feature": node["feat"],
                "name": self.encoder.headers[node["feat"]],
                "relation": "==",
                "reference": 1,
                "true": self.__translate__(node["left"]),
                "false": self.__translate__(node["right"])
            }
        else:
            raise Exception("Formatting Error: {}".format(str(node)))

    def predict(self, X):
        return self.tree.predict(X)

    def error(self, X, y, weight=None):
        return self.tree.error(X, y, weight=weight)

    def score(self, X, y, weight=None):
        return self.tree.score(X, y, weight=weight)

    def confusion(self, X, y, weight=None):
        return self.tree.confusion(self.predice(X), y, weight=weight)

    def latex(self):
        return self.tree.latex()

    def json(self):
        return self.tree.json()

    def binary_features(self):
        return len(self.encoder.headers)

    def __len__(self):
        return len(self.tree)

    def leaves(self):
        return self.tree.leaves()

    def nodes(self):
        return self.tree.nodes()

    def max_depth(self):
        return self.tree.maximum_depth()

    def regularization_upperbound(self, X, y):
        return self.tree.regularization_upperbound(X, y)

    def features(self):
        return self.tree.features()
