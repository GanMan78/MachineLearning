
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

def PlayPredictor(path):
    data=pd.read_csv(path)
    print("Dataset loaded successfully with the size",len(data))
    
    #prepare data
    Features=["Weather","Temperature"]
    print("Feature names are: ",Features)
    
    Weather=data.Weather
    Temperature=data.Temperature
    Play=data.Play
    
    lobj=preprocessing.LabelEncoder()  #this LabelEncoder is used to convert string data into numerical format.
    
    WeatherX=lobj.fit_transform(Weather)
    TemperatureX=lobj.fit_transform(Temperature)
    Label=lobj.fit_transform(Play)
    
    print("Encoded weather is: ")
    print(WeatherX)
    print("Encoded temperature is: ")
    print(TemperatureX)
    
    features=list(zip(WeatherX,TemperatureX))
    
    #step 3
    obj=KNeighborsClassifier(n_neighbors=3)
    obj.fit(features,Label)
    
    print("Give the weather and temperature details: ")
    dt=[]
    weat=input()
    dt.append(weat)
    temp=input()
    dt.append(temp)
    print(dt)
    test=lobj.fit_transform(dt)
    print(test)
    
    #step 4
    output=obj.predict([[2,1]])
    
    if output==1:
        print("You can play")
    else:
        print("Don't play")
    
def main():
    print("_______________Marvellous Play Predictor______________")
    print("Enter the path of the file which contains dataset")
    #path=input()
    
    PlayPredictor("PlayPredictor.csv")
    
    
if __name__=="__main__":
    main()
