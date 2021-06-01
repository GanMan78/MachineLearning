# Import Packages
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure,show
from seaborn import countplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# ML Operation
def TitanicLogistic():
    Line="*"*60
    print("Inside Logistic Function")
    
    # Step 1 - Load Data
    Titanic_data=pd.read_csv("TitanicDataset.csv")
    print("First 5 records of dataset : ")
    print(Titanic_data.head())
    print("Total number of records are : ",len(Titanic_data))
    
    # Step 2 - Analyse the data
    print(Line)
    print("Visualization of survived and non-survided passengers")
    print(Line)
    figure()                                                            # To create the diagram
    countplot(data=Titanic_data,x="Survived").set_title("Survived vs Non-survived")     # What to show
    #show()                                                              # To show the diagram on screen
    
    # Graph Visualization
    print("Visualization according to gender")
    print(Line)
    figure()
    countplot(data=Titanic_data,x="Survived",hue="Sex").set_title("Visualization according to sex")
    #show()
    
    # Graph Visualization
    print("Visualization according to passenger class")
    print(Line)
    figure()
    countplot(data=Titanic_data,x="Survived",hue="Pclass").set_title("Visualization according to Pclass")
    #show()
    
    # Graph Visualization - Histogram
    print("Survived vs Nonsurvived based on age")
    figure()
    Titanic_data["Age"].plot.hist().set_title("Visualization according to age")
    #show()
    
    # Step 3 - Data Cleaning
    print(Line)
    Titanic_data.drop("zero", axis=1,inplace=True)
    print("Data after column removal")
    print(Titanic_data.head())

    # Data Wrangling
    Sex=pd.get_dummies(Titanic_data["Sex"])
    print(Sex.head())
    Sex=pd.get_dummies(Titanic_data["Sex"],drop_first=True)
    print("Sex column after updation")
    print(Sex.head())
    
    Pclass=pd.get_dummies(Titanic_data["Pclass"],drop_first=True)
    print(Pclass.head())
    
    # Concate Sex and Pclass field in our dataset
    Titanic_data=pd.concat([Titanic_data,Sex,Pclass],axis=1)
    print("Data after concatenation : ")
    print(Titanic_data.head())
    
    # Removing unneccesary fields
    Titanic_data.drop(["Sex","sibsp","Parch","Embarked"],axis=1,inplace=True)
    print(Titanic_data.head())
    
    # Divide the dataset into X and Y
    x=Titanic_data.drop("Survived",axis=1)
    y=Titanic_data["Survived"]
    
    # Split the data for training and testing purpose
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.5)
    
    Obj=LogisticRegression(max_iter=2000)
    
    # Step 4 - Train the dataset
    Obj.fit(xtrain,ytrain)
    
    # Step 5 - Testing
    Output=Obj.predict(xtest)
    
    print("Accuracy of the given dataset is : ",(accuracy_score(ytest,Output))*100)
    print("Confusion matrics is : ")
    print(confusion_matrix(ytest,Output))
    
# Entry point function
def main():
    print("Logistic Case Study")
    TitanicLogistic()

# Starter
if __name__=="__main__":
    main()