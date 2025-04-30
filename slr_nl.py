import matplotlib.pyplot as plt

x=[1,2,3,4,5]
y=[3,7,3,5,1]

n=len(x)
mean_x=sum(x)/n
mean_y=sum(y)/n

numerator=sum((x[i]-mean_x)*(y[i]-mean_y) for i in range(n))
denominator=sum((x[i]-mean_x)**2 for i in range(n))
m=numerator/denominator
b=mean_y-m*mean_x
print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")

def predict(x_val):
    return m*x_val + b

predicted_y=[predict(val) for val in x]

print("Predicted values: ")
for val in x: 
    print(f"x={val}, predicted y={predict(val)}")   

plt.scatter(x, y, color='blue', label='Actual Data Points')

plt.plot(x, predicted_y, color='red', label='Regression Line')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
