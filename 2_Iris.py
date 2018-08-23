from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.externals.six import StringIO
import pydot
from sklearn.metrics import accuracy_score
from sklearn.model_selection  import train_test_split

#generate decision tree
def generate_decisionTree():
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True, impurity=False)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph[0].write_pdf("charts\\iris.pdf")



#flower data
iris = load_iris()
print('Total Sample size  - {size} '.format(size=len(iris.data)))
print('Properties are - {feature} '.format(feature=iris.feature_names))
print('Labels are - {feature} '.format(feature=iris.target_names))

# this is because Classifier are like y = f(x)
# here is x is data(input)= feature and y is target(output)= label
x = iris.data
y = iris.target

#Divide Train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

#Classifiers
clrKN = KNeighborsClassifier()
clf = tree.DecisionTreeClassifier()

#fit the data in clasifiers
clf.fit(x_train, y_train)
clrKN.fit(x_train, y_train)

#for visualization
generate_decisionTree()

#Predict
outputTree = clf.predict(x_test)
outKN = clrKN.predict(x_test)

#Results
print('Test target - {test_target} '.format(test_target=y_test))
print('Predicted Output Decision Tree Classifier :  - {output} '.format(output=outputTree))
print('Predicted Output KNeighbors  Classifier : - {output} '.format(output=outKN))

# Prediction accuracy
print("Accuracy for Decision Tree Classifier: " + str(accuracy_score(y_test, outputTree)*100)+"%")
print("Accuracy for KNeighbors Classifier: " + str(accuracy_score(y_test, outKN)*100)+"%")

