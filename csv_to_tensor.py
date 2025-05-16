import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import pickle

# Configuration
INPUT_FILE = "data.csv"
OUTPUT_DIR = "tensor_data"
INPUT_WINDOW_SIZE = 30  # 480 minutes for input
TARGET_WINDOW_SIZE = 30  # 30 minutes for target

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the raw price data
try:
    prices = np.loadtxt(INPUT_FILE, delimiter=',')
    print(f"Loaded {len(prices)} price points from '{INPUT_FILE}'")
    print(f"First 5 prices: {prices[:5]}")
    print(f"Last 5 prices: {prices[-5:]}")
except Exception as e:
    raise ValueError(f"Error loading data file: {e}")

# Verify data quality
if len(prices) < (INPUT_WINDOW_SIZE + TARGET_WINDOW_SIZE):
    raise ValueError(f"Not enough data points. Need at least {INPUT_WINDOW_SIZE + TARGET_WINDOW_SIZE} points, got {len(prices)}")
if np.isnan(prices).any():
    raise ValueError("Data contains NaN values")

# Calculate how many windows we can create
num_windows = len(prices) - (INPUT_WINDOW_SIZE + TARGET_WINDOW_SIZE) + 1
print(f"Creating {num_windows} sliding windows of data")

# Prepare empty arrays for inputs and targets
X = np.zeros((num_windows, INPUT_WINDOW_SIZE))
y = np.zeros((num_windows, TARGET_WINDOW_SIZE))

# Create input/target pairs with sliding window
for i in range(num_windows):
    input_start = i
    input_end = i + INPUT_WINDOW_SIZE
    target_start = input_end
    target_end = target_start + TARGET_WINDOW_SIZE
    X[i] = prices[input_start:input_end]
    y[i] = prices[target_start:target_end]

# Debug print sample windows
print("\nSample input window (first 5 values):", X[0][:5])
print("Sample target window (first 5 values):", y[0][:5])

# Normalize both input and target data using StandardScaler
scaler = StandardScaler()
X_reshaped = X.reshape(-1, 1)
X_scaled = scaler.fit_transform(X_reshaped)
X_scaled = X_scaled.reshape(X.shape)

y_reshaped = y.reshape(-1, 1)
y_scaled = scaler.transform(y_reshaped)
y_scaled = y_scaled.reshape(y.shape)

# Verify scaling
print("\nAfter scaling:")
print("Input mean:", np.mean(X_scaled))
print("Input std:", np.std(X_scaled))
print("Sample scaled input (first 5 values):", X_scaled[0][:5])
print("Target mean:", np.mean(y_scaled))
print("Target std:", np.std(y_scaled))
print("Sample scaled target (first 5 values):", y_scaled[0][:5])

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# Verify tensors
print("\nTensor verification:")
print("Input tensor shape:", X_tensor.shape)
print("Target tensor shape:", y_tensor.shape)
print("Input tensor min/max:", X_tensor.min().item(), X_tensor.max().item())
print("Target tensor min/max:", y_tensor.min().item(), y_tensor.max().item())

# Save tensors
torch.save(X_tensor, os.path.join(OUTPUT_DIR, "inputs.pt"))
torch.save(y_tensor, os.path.join(OUTPUT_DIR, "targets.pt"))
# Save scaler as pickle
with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

print(f"\nSaved {num_windows} input/target pairs to '{OUTPUT_DIR}'")
print("Data preparation complete!")