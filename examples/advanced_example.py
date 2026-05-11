"""
advanced_example.py - Advanced KiteML usage with custom options.
"""

from sklearn.datasets import load_diabetes

from kiteml import train

# Load a regression dataset
diabetes = load_diabetes(as_frame=True)
df = diabetes.frame

# Train with custom parameters
result = train(
    data=df,
    target="target",
    problem_type="regression",
    test_size=0.3,
    scale=True,
    random_state=123,
    verbose=True,
)

# Inspect all model scores
print("\n📈 All Model Scores:")
for model_name, score in sorted(
    result.all_results.items(), key=lambda x: x[1] if isinstance(x[1], float) else -1, reverse=True
):
    print(f"  {model_name:25s} -> {score}")

# Print detailed metrics
print("\n📋 Detailed Metrics:")
for k, v in result.metrics.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")

# Save and reload
result.save_model("diabetes_model.pkl")
loaded_model = result.load_model("diabetes_model.pkl")
