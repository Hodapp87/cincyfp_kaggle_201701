#!/usr/bin/env python

import pandas
import sklearn.tree
import sklearn.cross_validation
import sklearn.linear_model
import sklearn.metrics

# Read in CSV data for training:
data = pandas.read_csv("../data/winequality-data.csv")

# Split into inputs & outputs, training & test:
X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(
    data.drop(["id", "quality"], axis=1), # input attributes
    data["quality"], # outputs
    test_size = 0.33, # 1/3 of data is for testing
    random_state = 12345)

# (See also: http://scikit-learn.org/stable/modules/cross_validation.html)

# Fit a decision tree on all training data:
tree = sklearn.tree.DecisionTreeRegressor()
tree = tree.fit(X_train, y_train)

# Compute training error and testing error on this:
print("sklearn decision tree, training error: %f" %
      sklearn.metrics.mean_absolute_error(y_train, tree.predict(X_train)))
print("sklearn decision tree, testing error: %f" %
      sklearn.metrics.mean_absolute_error(y_test, tree.predict(X_test)))

# Fit linear regression on all training data:
lr = sklearn.linear_model.LinearRegression()
lr.fit(X_train, y_train)

# Compute training error and testing error, this time on the linear
# regression model:
print("sklearn linear regression, training error: %f" %
      sklearn.metrics.mean_absolute_error(y_train, lr.predict(X_train)))
print("sklearn linear regression, testing error: %f" %
      sklearn.metrics.mean_absolute_error(y_test, lr.predict(X_test)))

# Read in CSV for solution input:
soln = pandas.read_csv("../data/winequality-solution-input.csv")

# Compute predictions for solution input for decision tree & linear
# regression:
soln_predict_tree = tree.predict(soln.drop("id", axis=1))
soln_predict_lr = lr.predict(soln.drop("id", axis=1))

# Format as a submission and write as a CSV:
submission = pandas.DataFrame(index = soln["id"],
                              data = {'quality': soln_predict_tree})
submission.to_csv("winequality-submission-tree.csv", index_label = "id")
submission = pandas.DataFrame(index = soln["id"],
                              data = {'quality': soln_predict_lr})
submission.to_csv("winequality-submission-lr.csv", index_label = "id")
