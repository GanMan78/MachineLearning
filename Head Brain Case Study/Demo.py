
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def MeanData(arr):
    size=len(arr)
    sum=0
    
    for i in range(size):
        sum=sum+arr[i]
        
    return(sum/size)

def MarvellousHeadBrain(Name):
    dataset=pd.read_csv(Name)
    print("Size of our dataset is:",dataset.shape)
    
    X=dataset["Head Size(cm^3)"].values
    Y=dataset["Brain Weight(grams)"].values
    
    print("Lenght of X:",len(X))
    print("Lenght of Y:",len(Y))
    
    Mean_X=0
    Mean_Y=0
    Mean_X=MeanData(X)
    Mean_Y=MeanData(Y)
    print(type(Mean_X))

    print("Mean of independent variable is", Mean_X)
    print("Mean of dependent variable is", Mean_Y)
    #m=(Sum(X-Xb)*(Y-Yb))/Sum(X-Xb)^2
    numerator=0
    denomenator=0
    
    for i in range(len(X)):
        numerator=numerator+int((X[i] - Mean_X)*(Y[i] - Mean_Y))
        denomenator=denomenator+(X[i]-Mean_X)**2
        
    m=numerator/denomenator
    print("Value of m is",m)
    
    c=Mean_Y-(m*Mean_X)
    print("Value of c is: ",c)
    
    X_Start=np.min(X)-200
    X_End=np.max(X)+200
    
    x=np.linspace(X_Start,X_End)
    y=m*x+c
    
    plt.plot(x,y,color='r',label="Line of Regression")
    plt.scatter(X,Y,color='b', label="Data Plot")
    
    plt.xlabel("Head size")
    plt.ylabel("Brain Weight")
    
    plt.legend()
    plt.show()

def main():
    
    #print("Enter the name of the dataset")
    #name=input()
    MarvellousHeadBrain("MarvellousHeadBrain.csv")

if __name__=="__main__":
    main()