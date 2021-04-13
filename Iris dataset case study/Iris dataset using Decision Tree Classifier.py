
from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score

def main():

    dataset=load_iris()
    print("Features of dataset")
    print(dataset.feature_names)
    
    print("Target names of dataset")
    print(dataset.target_names)

    index=[1,51,101]
    test_target=dataset.target[index]
    test_feature=dataset.data[index]
    
    train_target=np.delete(dataset.target,index)
    train_feature=np.delete(dataset.data,index,axis=0)
    
    obj=tree.DecisionTreeClassifier()
    obj.fit(train_feature,train_target)    
    result=obj.predict(test_feature)
    
    print("Result prediction by ML:",result)
    print("Result expected", test_target)
    
    Accuracy=accuracy_score(test_target,result)
    print("Accuracy of the decision tree classifier algorithm is:",Accuracy*100,"%")

if __name__=="__main__":
    main()
