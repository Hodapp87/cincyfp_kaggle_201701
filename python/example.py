#!/usr/bin/env python

import numpy
import pandas
import sklearn.tree

# Read in CSV data for training input:
train = pandas.read_csv("../data/winequality-data.csv")

# Likewise, read in CSV for solution input:
soln = pandas.read_csv("../data/winequality-solution-input.csv")

# Fit a decision tree on all attributes (ignoring id & quality),
# against quality:
tree = sklearn.tree.DecisionTreeRegressor()
tree = tree.fit(train.drop(["id", "quality"], axis=1), train["quality"])

# Compute predictions for solution input:
soln_predict = tree.predict(soln.drop("id", axis=1))

# Format as a submission and write as a CSV:
submission = pandas.DataFrame(index = soln["id"],
                              data = {'quality': soln_predict})
submission.to_csv("winequality-submission.csv", index_label = "id")
