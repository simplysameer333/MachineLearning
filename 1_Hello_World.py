from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.spatial import distance


# dummy method that calls scipy euclidean and retun the distance between them
def euclidean_distance(a, b):
    return distance.euclidean(a, b)

# Custom class for Classifier that uses euclidean distance with k=1
class CustomizeClassifier:

    # this is calculate the euclidean_distance between train data and test data
    def closest(self, instance):
        best_distance = euclidean_distance(instance, self.X_train[0])
        best_index = 0;
        for i in range(1, len(self.X_train)):
            dist = euclidean_distance(instance, self.X_train[i])
            if dist < best_distance:
                best_distance = dist
                best_index = i
        return self.Y_train[best_index]

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, X_test):
        predictions = [];
        for instance in X_test:
            label = self.closest(instance)
            predictions.append(label)
        return predictions;


# flower data
iris = load_iris()
print('Total Sample size  - {size} '.format(size=len(iris.data)))
print('Properties are - {feature} '.format(feature=iris.feature_names))
print('Labels are - {feature} '.format(feature=iris.target_names))

# this is because Classifier are like y = f(x)
# here is x is data(input)= feature and y is target(output)= label
x = iris.data
y = iris.target

# Divide Train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

# Classifiers
clrKN = KNeighborsClassifier()
custom_clf = CustomizeClassifier()

# fit the data in classifiers
custom_clf.fit(x_train, y_train)
clrKN.fit(x_train, y_train)

# Predict
outKN = clrKN.predict(x_test)
outputTree = custom_clf.predict(x_test)

# Results
print('Test target - {test_target} '.format(test_target=y_test))
print('Predicted Output Decision Tree Classifier :  - {output} '.format(output=outputTree))
print('Predicted Output KNeighbors  Classifier : - {output} '.format(output=outKN))

# Prediction accuracy
print("Accuracy for KNeighbors Classifier: " + str(accuracy_score(y_test, outKN) * 100) + "%")
print("Accuracy for Custom Classifier: " + str(accuracy_score(y_test, outputTree) * 100) + "%")