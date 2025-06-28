import os
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from dataset.data_loader import DataLoader as MyDataLoader
from src.KAN.KAN_Net import KAN_Net
from src.MLP.MLP_Net import MLP_Net
from src.utility import (
    train_one_epoch,
    evaluate_model,
    plot_model_vs_itself,
    visualize_performance,
    plot_workspace_split_from_splitdata
)

# 1) Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2) Load BENDING dataset (Ms)
dl_bend = MyDataLoader()
dl_bend.load_data(deformation="bending", trial_num=2)
data_bend = dl_bend.get_data()
X_bend_raw = torch.tensor(data_bend['actuation'], dtype=torch.float32)
Y_bend = torch.tensor(data_bend['markers'][:, -1, :], dtype=torch.float32)

# Normalize X feature-wise to [-1,1]
X_min, X_max = X_bend_raw.min(0)[0], X_bend_raw.max(0)[0]
X_bend = 2 * (X_bend_raw - X_min) / (X_max - X_min) - 1.0

# 3) Load TWISTING dataset (Md)
dl_twist = MyDataLoader()
dl_twist.load_data(deformation="twisting_CW", trial_num=2)
data_twist = dl_twist.get_data()
X_twist_raw = torch.tensor(data_twist['actuation'], dtype=torch.float32)
Y_twist = torch.tensor(data_twist['markers'][:, -1, :], dtype=torch.float32)

# Normalize TWISTING dataset using same bounds as BENDING
X_twist = 2 * (X_twist_raw - X_min) / (X_max - X_min) - 1.0

# Plot workspace split
plot_workspace_split_from_splitdata(Y_bend, Y_twist)

# 4) DataLoaders
batch_size = 64
epochs = 50
loader_bend = DataLoader(TensorDataset(X_bend, Y_bend), batch_size=batch_size, shuffle=True)
loader_twist_kan = DataLoader(ConcatDataset([
    TensorDataset(X_twist, Y_twist),
    TensorDataset(X_bend[:int(0.8 * X_bend.shape[0])], Y_bend[:int(0.8 * X_bend.shape[0])])
]), batch_size=batch_size, shuffle=True)
loader_twist_mlp = DataLoader(TensorDataset(X_twist, Y_twist), batch_size=batch_size, shuffle=True)

# 5) Prepare results dir
results_dir = "plots/results_bending_twisting_reg"
os.makedirs(results_dir, exist_ok=True)

# 6) Build models + optimizers
kan = KAN_Net(
    input_dim=9,
    layer_configs=[
        {"out_features": 64, "n_knots": 16, "x_min": -5.2, "x_max": 5.2, "use_bn": True, "dropout": 0.0},
        {"out_features": 32, "n_knots": 12, "x_min": -5.2, "x_max": 5.2, "use_bn": False, "dropout": 0.0},
    ],
    output_dim=3
).to(device)

mlp = MLP_Net(input_dim=9, hidden_dims=[64, 32], output_dim=3).to(device)

opt_kan = torch.optim.Adam(kan.parameters(), lr=1e-2)
opt_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-2)

loss_fn = nn.MSELoss()
model_names = ['KAN', 'MLP']
models = [kan, mlp]
opts = [opt_kan, opt_mlp]

# === Phase 1: Ms (train on BENDING) ===
print("\n=== Phase 1: Ms (train on BENDING) ===")
for epoch in range(1, epochs + 1):
    for name, model, opt in zip(model_names, models, opts):
        loss = train_one_epoch(model, loader_bend, loss_fn, opt, device)
        print(f"[MsDs][Epoch {epoch}/{epochs}] {name} train loss: {loss:.4f}")

# Save KAN coeffs for regularization
kan_old_coeffs = [block.kan.coeffs.detach().clone() for block in kan.blocks]

# Freeze first KAN block to preserve bending knowledge
for param in kan.blocks[0].parameters():
    param.requires_grad = False

# Visualize performance on MsDs
print("\nValutazione su MsDs (bending)...")
test_loader = DataLoader(TensorDataset(X_bend, Y_bend), batch_size=128)
visualize_performance([], [], kan, mlp, test_loader, device, name='MsDs')

# === Phase 2: Md (fine-tune on TWISTING with replay only for KAN) ===
print("\n=== Phase 2: Md (fine-tune on TWISTING) ===")
lambda_reg = 5e-3
for epoch in range(1, epochs + 1):
    for name, model, opt in zip(model_names, models, opts):
        model.train()
        total_loss = 0.0
        loader = loader_twist_kan if name == 'KAN' else loader_twist_mlp
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            opt.zero_grad()
            y_pred = model(x_batch)
            base_loss = loss_fn(y_pred, y_batch)
            if name == 'KAN':
                reg_loss = sum(((block.kan.coeffs - old_c.to(device)) ** 2).sum()
                               for block, old_c in zip(model.blocks, kan_old_coeffs))
                loss = base_loss + lambda_reg * reg_loss
            else:
                noisy_y = y_batch + 0.1 * torch.randn_like(y_batch)
                loss = loss_fn(y_pred, noisy_y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"[MdDd][Epoch {epoch}/{epochs}] {name} train loss: {avg_loss:.4f}")

# Visualize performance on MdDs
print("\nValutazione su MdDs (bending post-twisting)...")
test_loader = DataLoader(TensorDataset(X_bend, Y_bend), batch_size=128)
visualize_performance([], [], kan, mlp, test_loader, device, name='MdDs')

# Visualize performance on MdDd
print("\nValutazione su MdDd (twisting)...")
test_loader = DataLoader(TensorDataset(X_twist, Y_twist), batch_size=128)
visualize_performance([], [], kan, mlp, test_loader, device, name='MdDd')

# === Plot forgetting ===
for name in model_names:
    for metric in ['X_RMSE', 'Y_RMSE', 'Z_RMSE', 'X_R2', 'Y_R2', 'Z_R2']:
        plot_model_vs_itself(
            results_dir,
            model_name=name,
            metric=metric,
            labels=('MsDs', 'MdDs')
        )
