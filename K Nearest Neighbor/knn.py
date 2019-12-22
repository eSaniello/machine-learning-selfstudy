import pandas as pd
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import numpy as np

# importing the data set
data = pd.read_csv("car.data")
print(data.head())

# changing text based labels into integers
# Ex: bad = 0, med = 1, good = 2
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

# splitting the data into x and y
x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

# randomly removing a portion of the data set for training and testing the model
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# training the model
neighbors = 9
model = KNeighborsClassifier(n_neighbors=neighbors)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

# testing the model with new data
predicted = model.predict(x_test)

names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted:", names[predicted[x]], " Data:", x_test[x], " Actual:", names[y_test[x]])
    # printing all the nearest neighbors
    n = model.kneighbors([x_test[x]], neighbors, True)
    print("N:", n)