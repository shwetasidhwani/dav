import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load your dataset
df = pd.read_csv('Social_Network_Ads.csv')

# Select features and target
X = df[['Age', 'EstimatedSalary']].values
y = df['Purchased'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc)
print("Confusion Matrix:\n", cm)

# Plot decision boundary
def plot_boundary(X, y, model):
    x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
    y_min, y_max = X[:, 1].min()-1000, X[:, 1].max()+1000
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k')
    plt.xlabel("Age")
    plt.ylabel("Estimated Salary")
    plt.title("Logistic Regression Decision Boundary")
    plt.grid(True)
    plt.show()

plot_boundary(X, y, model)
