
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def WinePredictor():
    Line="*"*70
    #Loading of dataset
    wine=datasets.load_wine()
    print(Line)    
    #Print features of data
    print("Features of wine")
    print(wine.feature_names)
    print(Line)
    
    #Print labels species(class_0, class_1, class_2)
    print("Labels of wines")
    print(wine.target_names)
    print(Line)
    
    #Print the wine labels(0:class_0, 1:class_1, 2:class_2)
    print("Targets of wines")
    print(wine.target)
    print(Line)
    
    #Printing first 5 rows of data
    print("First 5 rows of data")
    print(wine.data[0:5])
    print(Line)
    
    #Spliting of data into training and testing data set
    x_train,x_test,y_train,y_test=train_test_split(wine.data,wine.target,test_size=0.3) 
    
    #Create KNN Classifier
    knn=KNeighborsClassifier(n_neighbors=3)
    
    #Train the dataset
    knn.fit(x_train,y_train)
    
    #Predict the response for test dataset
    y_pred=knn.predict(x_test)
    
    #Accuracy of the model
    print("Accuracy of the model is:",metrics.accuracy_score(y_test,y_pred)*100)
    print(Line)
    
def main():
    Line="*"*70
    print("Machine Learning Application")
    print(Line)
    print("Wine Predictor Application using K Nearest Neighbor Algorithm")
    WinePredictor()
    
    
if __name__=="__main__":
    main()