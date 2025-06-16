import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from dataset.data_loader import DataLoader as MyDataLoader
from src.KAN.KAN_Net import KAN_Net
from src.MLP.MLP_Net import MLP_Net
from src.utility import (
    split_dataset_by_tip_position, evaluate_model,
    plot_generalization_results, plot_model_vs_itself
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Load dataset ===
dl = MyDataLoader()
dl.load_data(deformation="bending", trial_num=2)
data = dl.get_data()
X = torch.tensor(data['actuation'], dtype=torch.float32)
Y = torch.tensor(data['markers'][:, -1, :], dtype=torch.float32)
splits = split_dataset_by_tip_position(X, Y)
X_left, Y_left = splits['left']
X_right, Y_right = splits['right']

# === Common settings ===
batch_size = 64
epochs = 100
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)


# === Define models ===
def build_models():
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

    mlp = MLP_Net(input_dim=9, hidden_dims=[64, 32], output_dim=3).to(device)
    return kan, mlp


loss_fn = nn.MSELoss()


# === Train function ===
def train(model, optimizer, loader, loss_list):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    loss_list.append(total_loss / len(loader.dataset))


# === Evaluate function ===
def evaluate(models, X, Y, label):
    preds = {"true": Y.numpy()}
    records = []

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            y_pred = model(X.to(device)).cpu().numpy()
            preds[f"pred_{name}"] = y_pred
            metrics = evaluate_model(model, X, Y, device)
            records.append({"side": label, "model": name, **metrics})
            print(f"[{label}] {name}:", metrics)

    # Save
    csv_path = os.path.join(results_dir, f"evaluation_{label}.csv")
    npz_path = os.path.join(results_dir, f"predictions_{label}.npz")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    np.savez(npz_path, **preds)


# === Phase 1: MsDs (train LEFT, test LEFT) ===
kan, mlp = build_models()
opt_kan = torch.optim.Adam(kan.parameters(), lr=1e-3)
opt_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-3)
train_loader_left = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_left, Y_left), batch_size=batch_size,
                                                shuffle=True)

train_losses_kan, train_losses_mlp = [], []
for _ in range(epochs):
    train(kan, opt_kan, train_loader_left, train_losses_kan)
    train(mlp, opt_mlp, train_loader_left, train_losses_mlp)

evaluate({'KAN': kan, 'MLP': mlp}, X_left, Y_left, label="MsDs")

# === Phase 2: MdDs (train RIGHT, test LEFT) ===
kan2, mlp2 = build_models()
opt_kan2 = torch.optim.Adam(kan2.parameters(), lr=1e-3)
opt_mlp2 = torch.optim.Adam(mlp2.parameters(), lr=1e-3)
train_loader_right = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_right, Y_right),
                                                 batch_size=batch_size, shuffle=True)

for _ in range(epochs):
    train(kan2, opt_kan2, train_loader_right, train_losses_kan)
    train(mlp2, opt_mlp2, train_loader_right, train_losses_mlp)

evaluate({'KAN': kan2, 'MLP': mlp2}, X_left, Y_left, label="MdDs")

# === Phase 3: MdDd (test RIGHT) ===
evaluate({'KAN': kan2, 'MLP': mlp2}, X_right, Y_right, label="MdDd")

# === Plot generalization ===
for label in ["MsDs", "MdDs", "MdDd"]:
    plot_generalization_results(
        results_csv=os.path.join(results_dir, f"evaluation_{label}.csv"),
        semipiani_npz=os.path.join(results_dir, f"predictions_{label}.npz"),
        train_losses_kan=train_losses_kan,
        test_losses_kan=None,
        train_losses_mlp=train_losses_mlp,
        test_losses_mlp=None,
        results_dir=results_dir
    )

# === Plot KAN vs KAN and MLP vs MLP: MsDs vs MdDs ===
for model in ['KAN', 'MLP']:
    for metric in ['X_RMSE', 'Y_RMSE', 'Z_RMSE', 'X_R2', 'Y_R2', 'Z_R2']:
        plot_model_vs_itself(results_dir, model_name=model, metric=metric, labels=('MsDs', 'MdDs'))
