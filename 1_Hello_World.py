from sklearn import tree
import pydot
from io import StringIO

##nput
##-"smooth" and #1-"bumpy"
features = [[140, 0], [130, 0], [150, 1], [170, 1], [120, 0], [175, 1], [185, 1], [200, 1]]

#output
labelsDic = {0: "apple", 1: "oranges", 2: "watermelon"}
labels = [0, 0, 1, 1, 0, 1, 1, 2]

#Classifier (input - > output)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

#visualize Decision Tree
dotfile = StringIO()
tree.export_graphviz(clf, out_file=dotfile)
graph = pydot.graph_from_dot_data(dotfile.getvalue())
graph[0].write_pdf("charts\\fruits.pdf")
dotfile.close()

result = clf.predict([[160, 0]])
print(labelsDic[result[0]])
