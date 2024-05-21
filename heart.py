import pandas as pd
import seaborn as sns
import numpy as np
# import matplotlib
from matplotlib import pyplot as plt
import keras
from keras import models
from keras import layers

def acc_chart(results):
    plt.title("accuracy Graph")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.plot(results.history['accuracy'])
    plt.plot(results.history['val_accuracy'])
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

def loss_chart(results):
    plt.title("Loss Graph")
    plt.ylabel("Loss")
    plt.xlabel("epoch")
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

def do_graph_stuff(dfHeart):
    # Make a DataFrame copy for Male and graph it
    dfMale = dfHeart[dfHeart['sex'] == "Male"]
    sns.countplot(x='condition', data=dfMale)
    plt.title("Ratios of Healthy Males to Heart Risk")
    plt.show()

    # Make a DataFrame copy for Female and graph it
    dfFemale = dfHeart[dfHeart['sex'] == "Female"]
    sns.countplot(x='condition', data=dfFemale)
    plt.title("Ratios of Healthy Females to Heart Risk")
    plt.show()

    # Let's graph the "condition" with respect to the age and the condition
    condHealth = dfHeart['condition'] == 'Healthy'
    condAtRisk = dfHeart['condition'] == 'Heart Risk'

    # Histograph
    # Compare age with Health
    plt.hist(dfHeart[condHealth]['age'], color='b', alpha=0.5, bins=15, label="Healthy")
    plt.hist(dfHeart[condAtRisk]['age'], color='g', alpha=0.5, bins=15, label="At Risk")
    plt.title("Age and Heart Risk")
    plt.legend()
    plt.show()

    # Exercise 2 - Compare cholesterol with Health
    plt.hist(dfHeart[condHealth]['chol'], color='b', alpha=0.5, bins=15, label="Healthy")
    plt.hist(dfHeart[condAtRisk]['chol'], color='g', alpha=0.5, bins=15, label="At Risk")
    plt.title("Cholesterol (chol) and Health")
    plt.legend()
    plt.show()

    # Exercise Pt 2
    # As part of this, identify all the entries in the outlier for Cholesterol
    # In other words, create a DataSet just composed of the High Cholesterol entries
    dfHG = dfHeart[dfHeart['chol'] > 500]
    print("\n-- Exercise Pt 2 Outlier Cholesterol Entry --")
    print(dfHG.head().to_string())

    # Exercise 3 - Compare Max Heart Rate with Health (thalach)
    plt.hist(dfHeart[condHealth]['thalach'], color='b', alpha=0.5, bins=15, label="Healthy")
    plt.hist(dfHeart[condAtRisk]['thalach'], color='g', alpha=0.5, bins=15, label="At Risk")
    plt.title("Max Heart Rate and Health Rick")
    plt.legend()
    plt.show()

    # Exercise Pt 3 - Figure out the outlier/ number from this DataSet for Max Heart Rate and Health
    df_low_high = dfHeart[dfHeart['thalach'] < 80]
    print("\n-- Exercise Pt 3 Outlier Thalach Entry --")
    print(df_low_high.head().to_string())



# Create a DataFrame
dfHeart = pd.read_csv("Data/heart_cleveland_upload.csv")

# Make a copy
dfCopy = dfHeart.copy()

# -- Enhance the copy --
dfCopy['condition'] = dfCopy['condition'].map({0: "Healthy", 1: "Heart Risk"})
# - This maps all the 'condition' values of 0's to be Healthy, and the 1's to be Heart Risk

dfCopy['sex'] = dfCopy['sex'].map({1: "Male", 0: "Female"})
# - This maps all the 'sex' values of 1's to be Female, and the 0's to be Male

print(dfCopy.head().to_string())

# Call to do_graph_stuff
#do_graph_stuff(dfCopy)


# Now we want to go back to the Main dfHeart, not the copy and remove the outliers
# So get rid of the entires that are not less than that
dfHeart = dfHeart[dfHeart['chol'] < 500]
dfHeart = dfHeart[dfHeart['thalach'] > 80]
# - These will remove the entries that we had to remove from our values

# Identify the X and Y (X is default for inputs, y is default for output)
# - How many inputs should I have? 13
# - What are we taking out of this? condition
X = dfHeart.drop('condition', axis=1)
Y = dfHeart['condition']

print("\nShape of X is %s" % str(X.shape))
print("Shape of Y is %s" % str(Y.shape))
# Output - 295 rows, 13 entries
# - This is a sequential problem because we use the numbers backwards


# -- Keras Stuff --
# Set up our Model
model = models.Sequential()

# Two Layer Models
# - Input layer Model
model.add(layers.Dense(13, activation='relu'))
# - Output Layer Model
model.add(layers.Dense(1, activation='sigmoid'))

# Binary Cross Entropy Loss - Used for Binary Problems, See Machine Learning PP in Notes
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, Y, validation_split=0.22, epochs=200)

# Map these out with graphs
acc_chart(history)
loss_chart(history)

X_at_Risk = np.array([[62,1,3,145,250,1,2,120,0,1.4,1,1,0]],dtype=np.float64)
print(model.predict(X_at_Risk))
Y_at_Risk = (model.predict(X_at_Risk) > 0.5).astype(int)
print(Y_at_Risk[0])

X_Healthy = np.array([[50,1,2,129,196,0,0,163,0,0,0,0,0]],dtype=np.float64)
Y_Healthy = (model.predict(X_Healthy) > 0.5).astype(int)
print(Y_Healthy[0])


