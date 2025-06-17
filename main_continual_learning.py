import os
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from dataset.data_loader import DataLoader as MyDataLoader
from src.KAN.KAN_Net import KAN_Net
from src.MLP.MLP_Net import MLP_Net
from src.utility import (
    split_dataset_by_tip_position,
    train_one_epoch,
    evaluate_and_save,
    evaluate_model,
    plot_model_vs_itself
)

# 1) Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2) Load & normalize dataset
dl = MyDataLoader()
dl.load_data(deformation="bending", trial_num=2)
data = dl.get_data()
X_raw = torch.tensor(data['actuation'], dtype=torch.float32)
Y = torch.tensor(data['markers'][:, -1, :], dtype=torch.float32)

# Normalize X feature‐wise to [-1,1]
X_min, X_max = X_raw.min(0)[0], X_raw.max(0)[0]
X = 2 * (X_raw - X_min) / (X_max - X_min) - 1.0

# 3) Split semipiani based on tip position along Y (axis=1) w.r.t. 0.0
splits = split_dataset_by_tip_position(X, Y, axis=1, threshold=2.0)
X_left, Y_left = splits['left']
X_right, Y_right = splits['right']

# 4) DataLoaders
batch_size = 64
epochs = 100
loader_left = DataLoader(TensorDataset(X_left, Y_left), batch_size=batch_size, shuffle=True)
loader_right = DataLoader(TensorDataset(X_right, Y_right), batch_size=batch_size, shuffle=True)

# 5) Prepare results dir
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# 6) Build models + optimizers
kan = KAN_Net(
    input_dim=9,
    layer_configs=[
        {"out_features": 64, "n_knots": 16, "x_min": -1.0, "x_max": 1.0, "use_bn": True, "activation": nn.ReLU(),
         "dropout": 0.1},
        {"out_features": 32, "n_knots": 12, "x_min": -1.0, "x_max": 1.0, "use_bn": False, "activation": nn.ReLU(),
         "dropout": 0.0},
    ],
    output_dim=3
).to(device)

mlp = MLP_Net(input_dim=9, hidden_dims=[32, 32], output_dim=3).to(device)

opt_kan = torch.optim.Adam(kan.parameters(), lr=1e-3)
opt_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-3)

loss_fn = nn.MSELoss()
model_names = ['KAN', 'MLP']
models = [kan, mlp]
opts = [opt_kan, opt_mlp]

# === Phase 1: Ms (train on LEFT) ===
print("\n=== Phase 1: Ms (train on LEFT) ===")
for epoch in range(1, epochs + 1):
    for name, model, opt in zip(model_names, models, opts):
        loss = train_one_epoch(model, loader_left, loss_fn, opt, device)
        print(f"[MsDs][Epoch {epoch}/{epochs}] {name} train loss: {loss:.4f}")

# Evaluate Ms on LEFT
evaluate_and_save(
    {'KAN': kan, 'MLP': mlp},
    X_left, Y_left,
    device, evaluate_model, results_dir,
    label='MsDs'
)

# === Phase 2: Md (fine‐tune on RIGHT) ===
print("\n=== Phase 2: Md (fine‐tune on RIGHT) ===")
for epoch in range(1, epochs + 1):
    for name, model, opt in zip(model_names, models, opts):
        loss = train_one_epoch(model, loader_right, loss_fn, opt, device)
        print(f"[MdDs][Epoch {epoch}/{epochs}] {name} train loss: {loss:.4f}")

# Evaluate Md on LEFT (MdDs)
evaluate_and_save(
    {'KAN': kan, 'MLP': mlp},
    X_left, Y_left,
    device, evaluate_model, results_dir,
    label='MdDs'
)

# Evaluate Md on RIGHT (MdDd)
evaluate_and_save(
    {'KAN': kan, 'MLP': mlp},
    X_right, Y_right,
    device, evaluate_model, results_dir,
    label='MdDd'
)

# === Plot Ms vs Md for each model ===
for name in model_names:
    for metric in ['X_RMSE', 'Y_RMSE', 'Z_RMSE', 'X_R2', 'Y_R2', 'Z_R2']:
        plot_model_vs_itself(
            results_dir,
            model_name=name,
            metric=metric,
            labels=('MsDs', 'MdDs')
        )
