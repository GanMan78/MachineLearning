
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Machine Learning Algorithm
def PredDiabetes():
    # Header
    print("---------Diabetes Predictor Using Decision Tree--------------")
    
    # Reading Dataset
    diabetes=pd.read_csv("diabetes.csv")

    # Displaying Data using different methods
    print("Columns of dataset : ")
    print(diabetes.columns)

    print("First 5 records of dataset : ")
    print(diabetes.head())

    print("Dimensions of diabetes data : {}".format(diabetes.shape))

    # Spliting of data into parts
    x_train,x_test,y_train,y_test=train_test_split(diabetes.loc[:,diabetes.columns!='Outcome'],diabetes['Outcome'],stratify=diabetes['Outcome'],random_state=66)

    # Displaying upeer 5 rows of training and testing data
    print(x_train.head())
    print(x_test.head())
    print(y_train.head())

    # Selection of algorithm
    tree=DecisionTreeClassifier(random_state=0)

    # Training of dataset
    tree.fit(x_train,y_train)

    # Displaying the accuracy of training and testing dataset
    print("Accuracy on training set: {:.3f}".format(tree.score(x_train,y_train)))   # Displaying accuracy and rounding of points upto 3 decimals (:.3f)
    print("Accuracy on test set: {:.3f}".format(tree.score(x_test,y_test)))

    # Displaying feature importance of dataset
    print("Feature importances:\n{}".format(tree.feature_importances_))
   
    # Defining data plot function
    def plot_feature_importances_diabetes(model):
        plt.figure(figsize=(8,6))   # Defining fig size
        n_features=8
        plt.barh(range(n_features),model.feature_importances_,align="center")  # Horizontal bar graph
        diabetes_features=[x for i,x in enumerate(diabetes.columns) if i!=8]
        plt.yticks(np.arange(n_features),diabetes_features)
        plt.xlabel("Feature importance")
        plt.ylabel("Feature")
        plt.ylim(-1,n_features)
        plt.show()
    
    # Data Visualization - Displaying plot
    plot_feature_importances_diabetes(tree)

def main():
    PredDiabetes()   
    
if __name__=="__main__":
    main()

