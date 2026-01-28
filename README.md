# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: 
RegisterNumber:  
*/
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Dataset: marks vs result
X = np.array([[30], [35], [40], [45], [50], [55], [60], [65]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])   # 0 = Fail, 1 = Pass

# Create and train model
model = LogisticRegression()
model.fit(X, y)

# Predict for a new student
marks = [[48]]
result = model.predict(marks)
print("Prediction (0=Fail, 1=Pass):", result)

# Plot data points
plt.scatter(X, y, color='red', label='Actual Data')

# Plot logistic regression curve
X_test = np.linspace(25, 70, 100).reshape(-1, 1)
plt.plot(X_test, model.predict_proba(X_test)[:,1], label='Prediction Curve')

plt.xlabel("Marks")
plt.ylabel("Probability of Pass")
plt.title("Logistic Regression: Pass / Fail Prediction")
plt.legend()
plt.show()
```

## Output:
<img width="822" height="587" alt="image" src="https://github.com/user-attachments/assets/c98326c1-f8b0-4f5e-a903-fb4ab356aa1f" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
