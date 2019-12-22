import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# importing the data set
data = pd.read_csv("student-mat.csv", sep=";")

# excluding the unnecessary data from the dataset
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

# splitting the data in x and y.
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# randomly removing a portion of the data set for training and testing the model
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# training the model and saving the best one
# best = 0
# for _ in range(1000):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#
#     linear = linear_model.LinearRegression()
#     linear.fit(x_train, y_train)
#     acc = linear.score(x_test, y_test)
#     print(acc)
#
#     if acc > best:
#         best = acc
#         with open("student_model.pickle", "wb") as f:
#             pickle.dump(linear, f)

# loading the saved model
pickle_in = open("student_model.pickle", "rb")
linear = pickle.load(pickle_in)

print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

# testing the model with new data
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final grade")
pyplot.show()