"""
basic_example.py - Simple KiteML usage example.
"""

from sklearn.datasets import load_iris

from kiteml import train

# Load the Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Train with a single function call
result = train(df, target="target")

# View the summary
print(result.summary())

# Save the model
result.save_model("iris_model.pkl")
