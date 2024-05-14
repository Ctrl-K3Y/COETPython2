import pandas as pd
myCSV = pd.read_csv("Data/wwIIAircraft.csv")
# print(myCSV)

# Lots of info = Let's only print our the first 10 entries

# print(myCSV.head(10))
print(myCSV.tail(10))

print(myCSV.dtypes)
