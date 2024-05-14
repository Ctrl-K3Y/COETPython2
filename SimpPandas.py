import numpy as np
import pandas as pd

series1 = pd.Series([12, 15, 17, 19, 20], index=["one", "two", "three", "four", "five"])
print(series1)

workout = {"Mon": "Legs", "Tues": "Core", "Web": "Biceps", "Thur": "Rest", "Fri": "Legs"}
sWorkout = pd.Series(workout)
print(sWorkout)

#  Cool thing - let's create a date range from June until Dec of this year
dates = pd.date_range("20240601", periods=6, freq="MS")  # MS = Month Start, ME = Months End
print(dates)

df2 = pd.DataFrame(
    {
        "A": 1.0,
        "B": pd.Timestamp("20240514"),
        "C": pd.Series([2, 3, 14, 19]),
        "D": np.array([3] * 4, dtype='int32'),
        "E": pd.Categorical(["test", "train", "test", "train"]),
        "F": "foo"
    }
)
print(df2)
print(df2.dtypes)

print("\nTwo Rows")
print(df2.loc[1:2])

print("\nRows after 1")
print(df2.loc[2:])

print("\n Rows at the end")
print(df2.loc[-2:])

# When accessing an individual element, specify the column as the second element
# This example wll give the column values for E
print(df2.loc[0:])


print("\nColumn values for E")
print(df2.loc[0:,"E"])
print("\nSingle element")

print(df2.loc[0:0,"E"])
print(df2.loc[0:1,"E"])
df2["G"] = pd.Series([9,11,12,14])
print("After addition")
print(df2)
