import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib as plt

df = pd.read_csv("Data/kc_house_data.csv")
# print(df.head().to_string())
print("\n Shape and size")
print(df.shape)

print(df.loc[:0, "date"])

# Get the year out of this
# Create a new series that will be called registration year and for our purposes
# we want this to be an integer
df['reg_year'] = df['date'].str[:4]


# We want to convert this to be an integer
df['reg_year'] = df['reg_year'].astype('int')
# print(df.head().to_string())
# print(df.dtypes)


# We want to create a new series (house_age) that will
# be the difference between the reg_year and either the yr_built
# of the yr_renovated

# Create a new series - initially set all values to be NaN
df['house_age'] = np.NAN

for i,j in enumerate(df['yr_renovated']): # Goes thorugh all the values in yr renovations and sign those as the j values
    if [j == i]:
        df.loc[i:i, 'house_age'] = df.loc[i:i, 'reg_year'] - df.loc[i:i, 'yr_built']
    else:
        df.loc[i:i, 'house_age'] = df.loc[i:i, 'reg_year'] - df.loc[i:i, 'yr_renovated']

# print(df.head().to_string())

df.drop(["date","yr_built","yr_renovated","reg_year"], axis=1, inplace=True)
df.drop(["zipcode","lat","long","id"], axis=1, inplace=True)

# print(df.dtypes)
# print(df.head().to_string())

# Normally you would have to set up a series of tests up to see if there are any odd values in our
# existing DataSet - The Authors for this example pointer out that some of the house are ages -1
# That is a bad data value

df_bad = df[df['house_age'] < 0]
print("\n\n Bad Data Points")
# print(df_bad.to_string())

df = df[df['house_age'] >= 0]
print(df.head().to_string())

# for i in df.columns:
#     sb.displot(df[i])
#     plt.pyplot.show()



# Map all of the series against each other
# sb.pairplot(df)
# plt.pyplot.show()

# plt.pyplot.figure(figsize=(20,10))
# sb.heatmap(df.corr(), annot=True)
# plt.pyplot.show()


# We want to train our model to calculate the
# expected price built upon the other parameters
# For this purpose we wamt to specify our inputs
# anything that is not the price
# We want to output to be the price
# By convention C is supposed to be the inputs
# Y is supposed to be the array of outputs

X = df.drop('price',axis=1)
Y = df['price']

print (X.head().to_string())
print (Y.head().to_string())
