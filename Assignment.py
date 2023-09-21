import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("C:\\Users\\Dell1\Downloads\\SwedishMotorInsurance.csv")
print(dataset)
x=dataset.iloc[:,-2:]
df=pd.DataFrame(x)
df.replace(0, np.nan, inplace=True)
x=df.values
y=dataset.iloc[:,-1].values
temp=dataset.iloc[:,-3].values
print("_______________________________________")
print(temp)
print("_______________________________________")

print(x)
print("____________")
#print(y)

#print("_______________________________")



from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, -2:])
x[:, -2:] = imputer.transform(x[:, -2:])
print(x)
print("___________________")
y=x[:,-1]
x=x[:,-2]
print(y)
print("_________________")

print(x)

x = np.hstack((temp.reshape(-1, 1), x.reshape(-1, 1)))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#model.score(x_test, y_test)



from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

print(y_pred)

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
#plt.title('Salary vs Experience (Training set)')
#plt.xlabel('Years of Experience')
#plt.ylabel('Salary')
plt.show()


model.score(x_test, y_test)
