
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

#Using K Nearest Neighbor
def IrisKNN():
    dataset=load_iris()
    
    data=dataset.data
    target=dataset.target
    
    data_train,data_test,target_train,target_test=train_test_split(data,target,test_size=0.5)
    
    cobj=KNeighborsClassifier()
    
    cobj.fit(data_train,target_train)
    output=cobj.predict(data_test)
    
    Accuracy=accuracy_score(target_test,output)
    return Accuracy
 
# Using Decision Tree Classifier 
def IrisDecision():
    dataset=load_iris()
    
    data=dataset.data
    target=dataset.target
    
    data_train,data_test,target_train,target_test=train_test_split(data,target,test_size=0.5)
    
    cobj=tree.DecisionTreeClassifier()
    
    cobj.fit(data_train,target_train)
    output=cobj.predict(data_test)
    
    Accuracy=accuracy_score(target_test,output)
    return Accuracy

def main():
    ret=IrisKNN()
    print("Accuracy of KNN algorithm is: ",ret*100,"%")
    
    ret=IrisDecision()
    print("Accuracy of decision tree algorithm is: ",ret*100,"%")
    

if __name__=="__main__":
    main()
