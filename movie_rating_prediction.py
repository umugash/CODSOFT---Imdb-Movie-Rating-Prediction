# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the movie dataset (replace 'your_dataset.csv' with your actual dataset)
file_path = r"C:\Users\umuga\Desktop\IMDb Movies India.csv\IMDb Movies India.csv"
movie_data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Display the column names
print("Column Names:", movie_data.columns)

# Convert 'Year' column to numeric (remove parentheses if present)
movie_data['Year'] = pd.to_numeric(movie_data['Year'], errors='coerce')

# Clean the 'Duration' column by removing non-numeric characters
movie_data['Duration'] = movie_data['Duration'].str.extract('(\d+)').astype(float)

# Based on the column names, update the 'features' variable
features = movie_data[['Year', 'Duration', 'Votes']]

# Target variable
target = movie_data['Rating']

# Handle missing values in the target variable
target.fillna(target.mean(), inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Visualize predictions vs. actual ratings
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual Ratings vs. Predicted Ratings")
plt.show()
