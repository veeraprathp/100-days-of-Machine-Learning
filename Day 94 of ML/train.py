import bentoml
 
 
from sklearn import datasets
from sklearn import svm
 
 #load the iris dataset
iris = datasets.load_iris()
X,y = iris.data,iris.target

#Train the model 

clf = svm.SVC(gamma='scale')
clf.fit(X,y)

#save the model
saved_model = bentoml.sklearn.save_model("iris_classifier",clf)
print(f"model saved: {saved_model}")


#