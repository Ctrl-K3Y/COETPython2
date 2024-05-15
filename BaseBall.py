import pandas as pd

pdBB = pd.read_csv("./Data/mlb_salaries.csv")
# create a DataFrame of only the players who played for toronto

print(pdBB.describe()) # gives information relating to values
print(pdBB.dtypes)
print(pdBB.head().to_string())
pdTOR = pdBB[pdBB["teamid"] == "TOR"]
print(pdTOR.to_string())
print(pdTOR.aggregate("count"))
print("\n\nTotal salary for Toronto")

print(pdTOR["salary"].aggregate("average"))

print("\n Max salary for toronto")
print(pdTOR["salary"].aggregate("max"))
# Which of the players had a salary of 22 million
pdMax = pdTOR[["salary"]].aggregate("max")
maxSal = float(pdMax["salary"])

pdPlayer = pdTOR[pdTOR["salary"] == maxSal]["player_name"]

print("\n")
print(pdPlayer)
print(pdMax)
print("The player %s with the max salary is $%.f" % (pdPlayer.values[0], float(pdMax["salary"])))

# Count the number of players on the blue jays that bats right

batRight = pdTOR[pdTOR['bats'] == 'R'].aggregate("count")
print("The number of players that bats right is %d" % batRight["bats"])