import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import numpy as np
import plotly.graph_objects as go

# Dataset class remains the same
class ProteinDataset(Dataset):
    def __init__(self, file_path, max_len=256):
        self.max_len = max_len
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq, coords = self.data[idx]
        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(coords, dtype=torch.float)
        )

# Model architecture remains the same
class ProteinStructurePredictor(nn.Module):
    def __init__(self, vocab_size=21, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 3)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

def get_data_splits(dataset_path="cleaned_protein_data.pkl", max_len=256, batch_size=32):
    """Create train/val/test splits using random_split"""
    full_dataset = ProteinDataset(dataset_path, max_len=max_len)
    
    # Define splits (e.g., 70% train, 15% val, 15% test)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    # Create splits
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # for reproducibility
    )
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def evaluate_model(model, dataloader):
    """Calculate validation loss"""
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for seqs, coords in dataloader:
            seqs, coords = seqs.to(device), coords.to(device)
            outputs = model(seqs)
            total_loss += criterion(outputs, coords).item()
    
    return total_loss / len(dataloader)

def train_model(model, train_loader, val_loader, epochs=30, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # Training loop
        for seqs, coords in train_loader:
            seqs, coords = seqs.to(device), coords.to(device)
            
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, coords)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        if val_loader:
            val_loss = evaluate_model(model, val_loader)
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss/len(train_loader):.4f} | "
                  f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), "best_model.pth")
        else:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss/len(train_loader):.4f}")
    
    # Save final model if no validation
    if not val_loader:
        torch.save(model.state_dict(), "best_model.pth")


def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    all_rmsd = []
    with torch.no_grad():
        for seqs, coords in test_loader:
            seqs, coords = seqs.to(device), coords.to(device)
            outputs = model(seqs)
            
            # Convert to numpy for RMSD calculation
            preds = outputs.cpu().numpy()
            targets = coords.cpu().numpy()
            
            for i in range(preds.shape[0]):
                rmsd = compute_rmsd(preds[i], targets[i])
                all_rmsd.append(rmsd)
    
    avg_rmsd = np.mean(all_rmsd)
    print(f"\nTest RMSD: {avg_rmsd:.3f} Å")
    return avg_rmsd

def compute_rmsd(pred, true):
    mask = np.all(true != 0, axis=-1)  # Only non-padded positions
    pred = pred[mask]
    true = true[mask]
    if len(pred) == 0:
        return 0.0
    return np.sqrt(np.mean(np.sum((pred - true)**2, axis=-1)))

# Prediction and Visualization Class (NEW)
class ProteinVisualizer:
    def __init__(self, model_path="best_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ProteinStructurePredictor().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def _encode_sequence(self, sequence, max_len):
        """Convert AA sequence to numerical indices"""
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        aa_to_idx = {aa: i+1 for i, aa in enumerate(amino_acids)}  # 0 = padding
        
        # Convert sequence to uppercase and filter valid AAs
        filtered_seq = [aa for aa in sequence.upper() if aa in aa_to_idx]
        encoded = [aa_to_idx[aa] for aa in filtered_seq]
        
        # Pad/truncate
        if len(encoded) < max_len:
            encoded += [0] * (max_len - len(encoded))
        else:
            encoded = encoded[:max_len]
        return encoded
    
    def predict_from_sequence(self, sequence, max_len=256):
        """Full pipeline from sequence to 3D coordinates"""
        encoded = self._encode_sequence(sequence, max_len)
        with torch.no_grad():
            coords = self.model(torch.tensor([encoded]).to(self.device))[0].cpu().numpy()
        return coords[:len(sequence)]  # Remove padding
    
    def interactive_3d_view(self, coordinates, sequence=None, save_html=None):
        """Interactive visualization with Plotly"""
        fig = go.Figure()
        
        # Add main backbone trace
        fig.add_trace(go.Scatter3d(
            x=coordinates[:,0], y=coordinates[:,1], z=coordinates[:,2],
            mode='lines+markers',
            marker=dict(size=4, color='blue'),
            line=dict(width=6, color='blue'),
            name='Predicted Backbone'
        ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        if save_html:
            fig.write_html(save_html)
        fig.show()


def run_training_pipeline():
    # Get data splits
    train_loader, val_loader, test_loader = get_data_splits()
    
    # Initialize model
    model = ProteinStructurePredictor()
    
    # Train
    train_model(model, train_loader, val_loader)
    
    # Test
    test_model(model, test_loader)
    
    return model
