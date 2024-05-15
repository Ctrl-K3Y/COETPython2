import pandas as pd
pdB = pd.DataFrame(
    {
        "Names" : pd.Series(["Paul","Ringo","George","John"]),
        "Status" : pd.Categorical(["Alive","Alive","Dead","Dead"]),
        "Plays" : pd.Series(["Sings","Drums","Base","Sings"]),
        "Band" : "Beatles"
    }
)

print(pdB)
print(f" pdB.loc[:1]: \n {pdB.loc[:1]}")
print(f" pdB.loc[2:2]: \n {pdB.loc[2:2]}")
print(f" pdB.loc[2]: \n {pdB.loc[2]}")
# print(f" pdB.head: \n {pdB.head(2)}") # also works

# Get all the singers in the band - if the category is "sings"
pdSings = pdB[pdB["Plays"]== "Sings"]
print("\n\n Singers")
print(pdSings)

print(pdB["Plays"])