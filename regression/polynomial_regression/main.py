import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv("Position_Salaries.csv")

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(x_train, y_train)
y_pred = lin_reg.predict(x_test)
np.set_printoptions(precision=2)


print(
    np.concatenate(
        (
            y_pred.reshape(len(y_pred), 1),
            y_test.reshape(len(y_test), 1),
        ),
        1,
    ),
)


from sklearn.preprocessing import PolynomialFeatures


pol_reg = PolynomialFeatures()
