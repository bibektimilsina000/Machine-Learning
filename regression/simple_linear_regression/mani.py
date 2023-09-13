from matplotlib.markers import MarkerStyle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# importing dataset
dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# splitting the dataset in train and test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# Tranning my model
from sklearn.linear_model import LinearRegression


regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)


# Visulizing tranning set result

plt.scatter(x_train, y_train, color="red")
plt.plot(x_train, regressor.predict(x_train), color="blue")
plt.title("Salary Vs Years of exp(Trainning)")
plt.xlabel("Years of Exp")
plt.ylabel("Salary")
plt.show()
# Visulizing Test set result

plt.scatter(x_test, y_test, color="red")
plt.plot(x_train, regressor.predict(x_train), color="blue")
plt.title("Salary Vs Years of exp (Test)")
plt.xlabel("Years of Exp")
plt.ylabel("Salary")
plt.show()
