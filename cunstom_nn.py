import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"timeseries_training_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

def save_plot(fig, name):
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved plot to {path}")

class PowerBiasActivation(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(size))  # Initialize to 1
        self.biases = nn.Parameter(torch.ones(size))   # Initialize to 1
        self.eps = 1e-6
    
    def forward(self, x):
        abs_x = torch.abs(x) + self.eps
        return torch.pow(abs_x, self.weights) - torch.pow(self.biases, self.biases)

class TimeSeriesPowerNN(nn.Module):
    def __init__(self, input_window_size=30, hidden_size=20):
        super().__init__()
        # First layer processes each time step independently
        self.fc1 = nn.Linear(1, hidden_size)
        self.act1 = PowerBiasActivation(hidden_size)
        
        # Second layer processes the hidden representations
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act2 = PowerBiasActivation(hidden_size)
        
        # Final layer predicts the target window
        self.fc3 = nn.Linear(hidden_size * input_window_size, 30)  # 30 is the target window size
        
        # Initialize linear layers
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        # x shape: (batch_size, input_window_size)
        batch_size = x.shape[0]
        
        # Process each time step through the first two layers
        x = x.unsqueeze(-1)  # (batch_size, input_window_size, 1)
        x = self.act1(self.fc1(x))  # (batch_size, input_window_size, hidden_size)
        x = self.act2(self.fc2(x))  # (batch_size, input_window_size, hidden_size)
        
        # Flatten and predict the target window
        x = x.reshape(batch_size, -1)  # (batch_size, input_window_size * hidden_size)
        return self.fc3(x)

def train_phase(model, device, train_loader, 
                train_weights=True, train_biases=True, 
                epochs=10, lr=0.01, phase_name=""):
    
    # Freeze/unfreeze parameters based on phase
    for name, param in model.named_parameters():
        if 'weights' in name:
            param.requires_grad = train_weights
        elif 'biases' in name:
            param.requires_grad = train_biases
        else:  # Linear layer parameters
            param.requires_grad = True
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 1000 == 0:
                print(f'{phase_name} Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        scheduler.step(epoch_loss/len(train_loader))
        losses.append(epoch_loss/len(train_loader))
        print(f'{phase_name} Epoch {epoch}, Avg Loss: {losses[-1]:.4f}')
    
    # Plot and save training curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses)
    ax.set_yscale('log')
    ax.set_title(f'{phase_name} Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    save_plot(fig, f"{phase_name.lower()}_loss")
    
    return losses

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the prepared tensors
    X_tensor = torch.load("tensor_data/inputs.pt")
    y_tensor = torch.load("tensor_data/targets.pt")
    
    print(f"Loaded input tensor shape: {X_tensor.shape}")
    print(f"Loaded target tensor shape: {y_tensor.shape}")
    
    # Create dataset and dataloader with smaller batch size
    dataset = TensorDataset(X_tensor, y_tensor)
    batch_size = 1024  # Reduced batch size to fit in memory
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = TimeSeriesPowerNN(input_window_size=30, hidden_size=20).to(device)
    
    # PHASE 1: Train only weights (biases fixed at 1)
    print("\n=== PHASE 1: Training weights only ===")
    phase1_loss = train_phase(
        model, device, train_loader,
        train_weights=True, 
        train_biases=False,
        epochs=5,  # Reduced epochs for demonstration
        phase_name="Weights_Only"
    )
    
    # PHASE 2: Train only biases (weights fixed)
    print("\n=== PHASE 2: Training biases only ===")
    phase2_loss = train_phase(
        model, device, train_loader,
        train_weights=False,
        train_biases=True,
        epochs=5,
        phase_name="Biases_Only"
    )
    
    # PHASE 3: Joint training
    print("\n=== PHASE 3: Joint training ===")
    phase3_loss = train_phase(
        model, device, train_loader,
        train_weights=True,
        train_biases=True,
        epochs=10,
        phase_name="Joint_Training"
    )
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        # Evaluate on a subset of the training set
        test_samples = 1000
        x_test = X_tensor[:test_samples].to(device)
        y_true = y_tensor[:test_samples].numpy()
        y_pred = model(x_test).cpu().numpy()
        
        # Plot some sample predictions
        for i in range(3):  # Plot first 3 examples
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(y_true[i], label='True Values')
            ax.plot(y_pred[i], label='Predicted', linestyle='--')
            ax.legend()
            ax.set_title(f'Time Series Prediction - Sample {i+1}')
            save_plot(fig, f"sample_prediction_{i+1}")
    
    # Save parameter evolution
    with open(os.path.join(output_dir, "training_summary.txt"), "w") as f:
        f.write("Training Summary:\n")
        f.write(f"Final weights-only loss: {phase1_loss[-1]:.6f}\n")
        f.write(f"Final biases-only loss: {phase2_loss[-1]:.6f}\n")
        f.write(f"Final joint training loss: {phase3_loss[-1]:.6f}\n")
        f.write("\nFinal Parameters:\n")
        f.write(f"Layer 1 weights: {model.act1.weights.data.cpu().numpy()}\n")
        f.write(f"Layer 1 biases: {model.act1.biases.data.cpu().numpy()}\n")
        f.write(f"Layer 2 weights: {model.act2.weights.data.cpu().numpy()}\n")
        f.write(f"Layer 2 biases: {model.act2.biases.data.cpu().numpy()}\n")

if __name__ == "__main__":
    main()
