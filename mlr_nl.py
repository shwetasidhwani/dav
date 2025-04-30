import matplotlib.pyplot as plt

X=[
    [1,2],
    [2,3],
    [4,3],
    [3,5],
    [5,4]
]
#list of input vectors with two features x1 & x2

Y=[5,6,10,11,13]
#y is the list of target/output values corresponding to each vector in X

n=len(X) #number of samples/data-points
num_features=len(X[0]) #number of features each input has (in this case 2)

means = [sum(x[i] for x in X)/n for i in range(num_features)] #a list of means for each feature across all samples
mean_y=sum(Y)/n #mean of the target/output values

coefficients = []
for j in range(num_features): 
    numerator = sum((X[i][j] - means[j]) * (Y[i] - mean_y) for i in range(n)) #numerator of the slope for feature j
    denominator = sum((X[i][j] - means[j])**2 for i in range(n)) #denominator of the slope for feature j
    coefficients.append(numerator/denominator) #slope for feature j

intercept = mean_y - sum(coefficients[j] * means[j] for j in range(num_features)) #intercept of the regression line

#y=b0+b1x1+b2x2
def predict(features): 
    return intercept + sum(coefficients[i] * features[i] for i in range(len(features))) #predicts the output value for a given input vector

print("Predicted values: ")
for i in range(n): 
    print(f"X={X[i]}, predicted Y={predict(X[i])}") #prints the predicted output value for each input vector in X


predicted_y = [predict(x) for x in X]

mse = sum((Y[i] - predicted_y[i]) ** 2 for i in range(n)) / n
rmse = mse ** 0.5
ss_total = sum((Y[i] - mean_y) ** 2 for i in range(n))
ss_res = sum((Y[i] - predicted_y[i]) ** 2 for i in range(n))
r2 = 1 - (ss_res / ss_total)

print("Intercept (b0):", intercept)
for i, coef in enumerate(coefficients):
    print(f"Coefficient b{i+1}: {coef}")
print(f"\nMSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R² Score: {r2:.3f}")

x1 = [row[0] for row in X]
x2_fixed = sum(row[1] for row in X) / n  # average x2 to fix it

x1_range = list(range(min(x1), max(x1) + 1))
regression_line = [predict([val, x2_fixed]) for val in x1_range]

# Scatter actual points (colored)
plt.scatter(x1, Y, color='blue', label='Actual Data')

# Plot regression line by fixing x2
plt.plot(x1_range, regression_line, color='red', label=f'Regression Line (x2={x2_fixed:.1f})')

plt.xlabel('x1')
plt.ylabel('y')
plt.title('Multiple Linear Regression (2D Projection)')
plt.legend()
plt.grid(True)
plt.show()