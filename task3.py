import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

# Read dataset
df = pd.read_csv("bank.csv", sep=";")

# Convert all text columns to numbers
encoder = LabelEncoder()

for column in df.columns:
    df[column] = df[column].astype(str)
    df[column] = encoder.fit_transform(df[column])

# Features and target
X = df.drop(columns=["y"])
y = df["y"]


print("First 5 rows of dataset:")
print(df.head())

print("\nPrediction target values:")
print(y.head())

print("\nModel trained successfully!")

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Plot graph
plt.figure(figsize=(15,8))
plot_tree(model, filled=True)

plt.title("Decision Tree Classifier")
plt.show()
