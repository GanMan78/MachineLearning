
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def MLHeadBrain(Name):
    dataset=pd.read_csv(Name)
    print("Size of our dataset is:",dataset.shape)
    
    X=dataset["Head Size(cm^3)"].values
    Y=dataset["Brain Weight(grams)"].values
    X=X.reshape(-1,1)
    obj=LinearRegression()
    obj.fit(X,Y)
    
    output=obj.predict(X)
    
    rsquare=obj.score(X,Y)
    
    print("Value of R square is: ",rsquare)
    
def main():
    
    #print("Enter the name of the dataset")
    #name=input()
    MLHeadBrain("HeadBrain.csv")

if __name__=="__main__":
    main()
