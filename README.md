# breast-cancer-detection
import pandas as pd
dataset=pd.read_csv("/content/breast-cancer.csv")
print(dataset)

dataset.drop_duplicates()

dataset.isnull()

dataset.info()

dataset.head()

dataset.tail()

dataset.isnull().sum()

dataset.backfill()

dataset.ffill

p=dataset.hist(figsize=(20,20))

dataset.columns

dataset.describe()

import matplotlib.pyplot as plt
plt.plot(dataset['compactness_mean'])
plt.xlabel("")
plt.ylabel("Thigh")
plt.title("Line Plot")
plt.show()

import matplotlib.pyplot as plt
plt.plot(dataset['concavity_mean'])
plt.xlabel("")
plt.ylabel("Thigh")
plt.title("Line Plot")
plt.show()

from sklearn import preprocessing
import pandas as pd
selected_columns = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean"]
d = preprocessing.normalize(dataset[selected_columns], axis=0)
scaled_df = pd.DataFrame(d, columns=selected_columns)
scaled_df.head()

import matplotlib.pyplot as plt
import pandas as pd
data = {
    'Result': [1, 1, 0, 0, 1, 0, 1],
    'perimeter_worst': [12, 15, 18, 20, 22, 25, 28]
}
dataset = pd.DataFrame(data)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
dataset_len_1 = dataset[dataset['Result'] == 1]["perimeter_worst"]
ax1.hist(dataset_len_1, color='red')
ax1.set_title('Having Breast Cancer')
ax1.set_xlabel('Perimeter Worst')
ax1.set_ylabel('Frequency')

dataset_len_0 = dataset[dataset['Result'] == 0]['perimeter_worst']
ax2.hist(dataset_len_0, color='green')
ax2.set_title('Not Having Breast Cancer')
ax2.set_xlabel('Perimeter Worst')
ax2.set_ylabel('Frequency')

fig.suptitle("Breast Cancer Levels")
plt.show()

dataset.duplicated()

from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report #for confusion matrix
from sklearn.linear_model import LogisticRegression,LinearRegression #logistic regression

train,test=train_test_split(dataset,test_size=0.3,random_state=0,stratify=dataset['Result'])
train_X=train[train.columns[:-1]]
train_Y=train[train.columns[-1:]]
test_X=test[test.columns[:-1]]
test_Y=test[test.columns[-1:]]
X=dataset[dataset.columns[:-1]]
Y=dataset['Result']
len(train_X), len(train_Y), len(test_X), len(test_Y)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
# Load the Breast Cancer Wisconsin dataset
data = load_breast_cancer()
features = data.feature_names
target = 'target'
# Create a DataFrame
df = pd.DataFrame(data.data, columns=features)
df[target] = data.target
# Split the dataset into train and test sets
train, test = train_test_split(df, test_size=0.3, random_state=0, stratify=df[target])
# Features (input variables)
features = df.columns[:-1]  # Exclude the target variable
# Standardize the features using StandardScaler
scaler = StandardScaler()
train_X = scaler.fit_transform(train[features])
test_X = scaler.transform(test[features])
# Target variable
train_Y = train[target]
test_Y = test[target]
# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(train_X, train_Y)
# Make predictions on the test set
prediction = model.predict(test_X)
# Evaluate the model
accuracy = metrics.accuracy_score(test_Y, prediction)
print('The accuracy of the Logistic Regression model is:', accuracy)
# Display classification report
report = classification_report(test_Y, prediction)
print("Classification Report:\n", report)

# Create and fit the Linear Regression model
model = LinearRegression()
model.fit(train_X, train_Y)
# Make predictions on the test set
prediction = model.predict(test_X)
# Assuming 'test_Y' contains the true labels for the test set
# Calculate the accuracy
accuracy = accuracy_score(test_Y, prediction.round())
# Print the accuracy
print('The accuracy of Linear Regression is:', accuracy)

# Import necessary libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
# ... (previous code for loading data, splitting, and fitting the model)
# Make predictions on the test data
y_pred = model.predict(X_test)
# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
# Calculate Root Mean Squared Error
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')
# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')

import matplotlib.pyplot as plt
# Values for logistic regression and linear regression
logistic_regression_value = 0.9590
linear_regression_value = 0.9415
# Y-axis values
y_values = [0.938, 0.941, 0.944, 0.947, 0.950, 0.953, 0.956, 0.959, 0.962]
# Bar positions
bar_positions = [1, 2]
# Bar heights
bar_heights = [logistic_regression_value, linear_regression_value]
# Bar labels
bar_labels = ['Logistic Regression', 'Linear Regression']
# Create bar graph
plt.bar(bar_positions, bar_heights, tick_label=bar_labels, color=['skyblue', 'skyblue'])
plt.ylim(0.935, 0.965)  # Set y-axis limits
# Add individual values above the bars
for i, value in enumerate(bar_heights):
    plt.text(bar_positions[i], value + 0.001, f'{value:.4f}', ha='center', va='bottom')
# Add labels and title
plt.xlabel('Regression Type')
plt.ylabel('Y-axis Values')
plt.title('Comparison of Logistic and Linear Regression Values')
# Show the plot
plt.show()

import matplotlib.pyplot as plt
import numpy as np
# Sample data
categories = ['Precision', 'Recall', 'F1 Score']
micro_avg = [0.96, 0.96, 0.96]  # Replace with your values
weighted_avg = [0.96, 0.96, 0.96]  # Replace with your values
# Bar graph
bar_width = 0.35
index = np.arange(len(categories))
plt.bar(index, micro_avg, bar_width, label='Micro Average')
plt.bar(index + bar_width, weighted_avg, bar_width, label='Weighted Average')
# Customize the plot
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Micro vs Weighted Averages for Precision, Recall, and F1 Score')
plt.xticks(index + bar_width / 2, categories)
plt.legend()
# Show the plot
plt.show()

import matplotlib.pyplot as plt
import numpy as np
# Sample data
metrics = ['MSE', 'RMSE', 'MAE', 'R-squared']
values = [0.1, 0.2, 0.15, 0.8]  # Replace with your actual values
# Bar graph
plt.bar(metrics, values, color=['skyblue', 'skyblue', 'skyblue', 'skyblue'])
# Customize the plot
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Performance Metrics')
plt.ylim(0, 1)  # Adjust the y-axis limits as needed
# Show the plot
plt.show()
