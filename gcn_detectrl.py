import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import os
from sklearn.metrics import roc_auc_score, f1_score

GRAPH_DATA_DIR = "graph_data"


TRAIN_FILE = "detectrl_train_split.pkl"
VAL_FILE = "detectrl_val.pkl"
TEST_FILE = "detectrl_test.pkl"


HIDDEN_DIM = 256  
OUTPUT_DIM = 64
EPOCHS = 15 
LEARNING_RATE = 0.001 
DROPOUT = 0.5 
SEED = 2024 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


print("Loading graph data...")
with open(os.path.join(GRAPH_DATA_DIR, TRAIN_FILE), "rb") as f:
    train_data = pickle.load(f)
with open(os.path.join(GRAPH_DATA_DIR, VAL_FILE), "rb") as f:
    val_data = pickle.load(f)
with open(os.path.join(GRAPH_DATA_DIR, TEST_FILE), "rb") as f:
    test_data = pickle.load(f)

print(f"Train samples: {len(train_data['y'])}")
print(f"Val samples: {len(val_data['y'])}")
print(f"Test samples: {len(test_data['y'])}")


print(f"\nTrain label distribution:")
print(f"  Human (0): {(train_data['y'] == 0).sum().item()}")
print(f"  AI (1): {(train_data['y'] == 1).sum().item()}")


class PRDetectGCN(nn.Module):
    """2-layer GCN as described in the paper (Section 3.2)"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PRDetectGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, 1)
        self.dropout = nn.Dropout(DROPOUT)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        # Global mean pooling (as in paper)
        x = torch.mean(x, dim=0, keepdim=True)
        return torch.sigmoid(x)

# ============ INITIALIZE MODEL ============
input_dim = 768  # RoBERTa embedding dimension
hidden_dim = HIDDEN_DIM
output_dim = OUTPUT_DIM

model = PRDetectGCN(input_dim, hidden_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

print(f"\nModel Architecture:")
print(f"  Input dim: {input_dim}")
print(f"  Hidden dim: {hidden_dim}")
print(f"  Output dim: {output_dim}")
print(f"  Dropout: {DROPOUT}")
print(f"  Total params: {sum(p.numel() for p in model.parameters())}")

# ============ TRAINING ============
train_len = len(train_data['y'])
val_len = len(val_data['y'])

train_losses = []
val_losses = []
train_accs = []
val_accs = []
best_val_acc = -1
early_stop_counter = 0

# TensorBoard writer
writer = SummaryWriter(f'logs/detectrl_seed_{SEED}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

print("\n" + "="*60)
print("Starting Training...")
print("="*60)

start_time = time.time()

for epoch in range(EPOCHS):
    # ===== TRAINING =====
    model.train()
    epoch_train_loss = 0.0
    correct_train = 0
    
    for i in tqdm(range(train_len), desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        data = Data(
            x=train_data['all_token_embeddings'][i],
            edge_index=train_data['all_edge_index'][i],
            y=train_data['y'][i]
        ).to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y.float().view(-1, 1))
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item()
        pred = (output >= 0.5).long()
        correct_train += (pred == data.y.view(-1, 1)).sum().item()
    
    avg_train_loss = epoch_train_loss / train_len
    train_acc = correct_train / train_len
    train_losses.append(avg_train_loss)
    train_accs.append(train_acc)
    
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Acc/train', train_acc, epoch)
    
    # ===== VALIDATION =====
    model.eval()
    epoch_val_loss = 0.0
    correct_val = 0
    
    with torch.no_grad():
        for i in tqdm(range(val_len), desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            data = Data(
                x=val_data['all_token_embeddings'][i],
                edge_index=val_data['all_edge_index'][i],
                y=val_data['y'][i]
            ).to(device)
            
            output = model(data)
            loss = criterion(output, data.y.float().view(-1, 1))
            
            epoch_val_loss += loss.item()
            pred = (output >= 0.5).long()
            correct_val += (pred == data.y.view(-1, 1)).sum().item()
    
    avg_val_loss = epoch_val_loss / val_len
    val_acc = correct_val / val_len
    val_losses.append(avg_val_loss)
    val_accs.append(val_acc)
    
    writer.add_scalar('Loss/val', avg_val_loss, epoch)
    writer.add_scalar('Acc/val', val_acc, epoch)
    
    print(f"\nEpoch {epoch+1}/{EPOCHS}:")
    print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs('model', exist_ok=True)
        torch.save(model.state_dict(), f'model/detectrl_gcn_model_seed_{SEED}.pth')
        print(f"   Best model saved! (Val Acc: {val_acc:.4f})")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= 3:  # Early stopping after 3 epochs no improvement
            print(f"  Early stopping at epoch {epoch+1}")
            break

training_time = time.time() - start_time
print(f"\n Training completed in {training_time:.2f} seconds")

# ============ TESTING ============
print("\n" + "="*60)
print("Testing on Test Set...")
print("="*60)

# Load best model
test_model = PRDetectGCN(input_dim, hidden_dim, output_dim).to(device)
test_model.load_state_dict(torch.load(f'model/detectrl_gcn_model_seed_{SEED}.pth'))
test_model.eval()

test_len = len(test_data['y'])
test_loss = 0.0
correct_test = 0
test_predictions = []

start_time = time.time()

with torch.no_grad():
    for i in tqdm(range(test_len), desc="Testing"):
        data = Data(
            x=test_data['all_token_embeddings'][i],
            edge_index=test_data['all_edge_index'][i],
            y=test_data['y'][i]
        ).to(device)
        
        output = test_model(data)
        test_predictions.append(output.item())
        
        loss = criterion(output, data.y.float().view(-1, 1))
        test_loss += loss.item()
        
        pred = (output >= 0.5).long()
        correct_test += (pred == data.y.view(-1, 1)).sum().item()

test_time = time.time() - start_time

# Calculate metrics
avg_test_loss = test_loss / test_len
test_acc = correct_test / test_len

# Convert to numpy for sklearn metrics
y_true = test_data['y'].cpu().numpy()
y_pred = [1 if p >= 0.5 else 0 for p in test_predictions]

test_f1 = f1_score(y_true, y_pred)
test_auc = roc_auc_score(y_true, test_predictions)

# Print results
print("\n" + "="*60)
print(" TEST RESULTS")
print("="*60)
print(f"Test Loss:     {avg_test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Test F1 Score: {test_f1:.4f}")
print(f"Test AUC:      {test_auc:.4f}")
print(f"Test Time:     {test_time:.2f} seconds")
print("="*60)

# Save results
os.makedirs('results', exist_ok=True)
with open(f"results/detectrl_results_seed_{SEED}.txt", "w", encoding="utf-8") as f:
    f.write(f"Dataset: DetectRL\n")
    f.write(f"Seed: {SEED}\n")
    f.write(f"Model: PRDetect GCN (2 layers)\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Learning Rate: {LEARNING_RATE}\n")
    f.write(f"Hidden Dim: {HIDDEN_DIM}\n")
    f.write(f"Dropout: {DROPOUT}\n")
    f.write(f"\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write(f"Test F1 Score: {test_f1:.4f}\n")
    f.write(f"Test AUC: {test_auc:.4f}\n")
    f.write(f"Test Loss: {avg_test_loss:.4f}\n")
    f.write(f"\n")
    f.write(f"Training time: {training_time:.2f}s\n")
    f.write(f"Testing time: {test_time:.2f}s\n")
    f.write(f"Date: {datetime.now()}\n")

print(f"\n Results saved to results/detectrl_results_seed_{SEED}.txt")

# Close tensorboard writer
writer.close()