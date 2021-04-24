
import pandas as pd
import matplotlib.pyplot as plt

def main():
    Line="*"*60
    excel='StudentData.xlsx'
    data=pd.read_excel(excel)
    print(Line)
    print("Size of the data is: ",data.shape)
    print("All data of excel sheet")
    print(Line)
    print(data)
    
    print(Line)
    print("First 5 rows from file")
    print(data.head())
    
    print(Line)
    print("Last 3 rows of file")
    print(data.tail(3))
    
    print(Line)
    sorted_data=data.sort_values(["Name"],ascending=False)
    print("Data after sorting according to reverse alphabetical order")
    print(sorted_data)
    
    #Plot 1
    data["Age"].plot(kind="hist")
    plt.xlabel("Age Band")
    plt.ylabel("No of Students")
    plt.title("Student-Age relationship using Histogram")
    plt.show()
    
    #Plot 2
    data["Age"].plot(kind="barh",color="m")
    plt.xlabel("Age of student")
    plt.ylabel("Students in order")
    plt.title("Student-Age relationship using horizontal Bar graph")
    plt.show()
    
    #Plot 3
    plt.plot(data["Name"],data["Age"], "g--", label='Default')
    plt.xlabel("Students")
    plt.ylabel("Age")
    plt.title("Student-Age relationship using Line plot")
    plt.legend(loc="best")
    plt.show()
    
    #Plot 4
    data["Age"].plot(kind="bar",color="c",alpha=0.7)
    plt.xlabel("Index wise Students")
    plt.ylabel("Age of student")
    plt.title("Student-Age relationship using vertical Bar graph")
    plt.show()
    
if __name__=="__main__":
    main()