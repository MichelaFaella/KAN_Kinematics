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
    plot_model_vs_itself, plot_workspace_split_from_splitdata
)

# 1) Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2) Load LEFT dataset (bending → Ds)
dl_left = MyDataLoader()
dl_left.load_data(deformation="bending", trial_num=2)
data_left = dl_left.get_data()
X_raw_left = torch.tensor(data_left['actuation'], dtype=torch.float32)
Y_left = torch.tensor(data_left['markers'][:, -1, :], dtype=torch.float32)

# 2b) Load RIGHT dataset (twisting → Dd)
dl_right = MyDataLoader()
dl_right.load_data(deformation="twisting_CW", trial_num=2)
data_right = dl_right.get_data()
X_raw_right = torch.tensor(data_right['actuation'], dtype=torch.float32)
Y_right = torch.tensor(data_right['markers'][:, -1, :], dtype=torch.float32)

# Calcolo normalizzazione su entrambe le sorgenti combinate
X_total = torch.cat([X_raw_left, X_raw_right], dim=0)
X_min, X_max = X_total.min(0)[0], X_total.max(0)[0]
X_left = 2 * (X_raw_left - X_min) / (X_max - X_min) - 1.0
X_right = 2 * (X_raw_right - X_min) / (X_max - X_min) - 1.0

plot_workspace_split_from_splitdata(Y_left, Y_right)

# 4) DataLoaders
batch_size = 64
epochs_phase1 = 200
epochs_phase2 = 30
loader_left = DataLoader(TensorDataset(X_left, Y_left), batch_size=batch_size, shuffle=True)
loader_right = DataLoader(TensorDataset(X_right, Y_right), batch_size=batch_size, shuffle=True)

# 5) Prepare results dir
results_dir = "plots/results"
os.makedirs(results_dir, exist_ok=True)

# 6) Build models + optimizers
kan = KAN_Net(
    input_dim=9,
    layer_configs=[
        {"out_features": 128, "n_knots": 32, "x_min": float(X_min[i]), "x_max": float(X_max[i]), "use_bn": True, "dropout": 0.0} for i in range(9)
    ] + [
        {"out_features": 64, "n_knots": 24, "x_min": -1.0, "x_max": 1.0, "use_bn": True, "dropout": 0.0}
    ],
    output_dim=3
).to(device)

mlp = MLP_Net(input_dim=9, hidden_dims=[64, 32], output_dim=3).to(device)

opt_kan = torch.optim.AdamW(kan.parameters(), lr=1e-3, weight_decay=1e-2)
opt_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-3)

loss_fn = nn.MSELoss()
model_names = ['KAN', 'MLP']
models = [kan, mlp]
opts = [opt_kan, opt_mlp]

# === Phase 1: Ms (train on Ds → bending) ===
print("\n=== Phase 1: Ms (train on Ds: bending) ===")
for epoch in range(1, epochs_phase1 + 1):
    for name, model, opt in zip(model_names, models, opts):
        loss = train_one_epoch(model, loader_left, loss_fn, opt, device)
        print(f"[MsDs][Epoch {epoch}/{epochs_phase1}] {name} train loss: {loss:.4f}")

# Evaluate Ms on Ds
evaluate_and_save(
    {'KAN': kan, 'MLP': mlp},
    X_left, Y_left,
    device, evaluate_model, results_dir,
    label='MsDs'
)

# === Phase 2: Md (fine‐tune on Dd → twisting) ===
print("\n=== Phase 2: Md (fine‐tune on Dd: twisting) ===")

# Reduce LR + freeze 1st layer of KAN to retain knowledge
for g in opt_kan.param_groups:
    g['lr'] = 5e-5
    g['weight_decay'] = 1e-2

if hasattr(kan, 'layers') and len(kan.layers) > 0:
    for param in kan.layers[0].parameters():
        param.requires_grad = False

for epoch in range(1, epochs_phase2 + 1):
    for name, model, opt in zip(model_names, models, opts):
        loss = train_one_epoch(model, loader_right, loss_fn, opt, device)
        print(f"[MdDd][Epoch {epoch}/{epochs_phase2}] {name} train loss: {loss:.4f}")

# Evaluate Md on Ds
evaluate_and_save(
    {'KAN': kan, 'MLP': mlp},
    X_left, Y_left,
    device, evaluate_model, results_dir,
    label='MdDs'
)

# Evaluate Md on Dd
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