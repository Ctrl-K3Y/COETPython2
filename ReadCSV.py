import pandas as pd
myCSV = pd.read_csv("Data/wwIIAircraft.csv")
# print(myCSV)

# Lots of info = Let's only print our the first 10 entries
print(myCSV.head(10).to_string())

# print(myCSV.head(10))
# print(myCSV.tail(10))
# print(myCSV.dtypes)

# List all the aircraft that were made by Germany
german_aircrafts = myCSV[myCSV["Country of Origin"] == "Germany"]
print(german_aircrafts.to_string())

pdGermany = myCSV.loc[(myCSV["Country of Origin"] == "Germany") &
                                    myCSV["Year in Service"] == 1942]
print("\n\n")
print(pdGermany.to_string())