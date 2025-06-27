import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from src.MLP.MLP_Net import MLP_Net
from src.KAN.KAN_Net import KAN_Net
from dataset.data_loader import DataLoader as MyDataLoader
from src.utility import plot_actuations_comp, plot_colored_trajectory_comp


# ----- Normalizzazione -----
def normalize(tensor):
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True) + 1e-8
    norm = (tensor - mean) / std
    return norm, mean.squeeze(), std.squeeze()

def denormalize(tensor, mean, std):
    return tensor * std + mean

# ----- Config -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using", device)

# ----- Load real dataset -----
dl = MyDataLoader()
dl.load_data(deformation="bending", trial_num=1)
data = dl.get_data()

actuations_raw = torch.tensor(data["actuation"], dtype=torch.float32)
positions_raw = torch.tensor(data["markers"][:, -1, :2], dtype=torch.float32)

# ----- Normalizzazione -----
actuations, mean_act, std_act = normalize(actuations_raw)
positions, mean_pos, std_pos = normalize(positions_raw)
print("✅ Normalizzazione completata")

# ----- Traiettorie sintetiche -----
def circle(radius, center, N):
    t = np.linspace(0, 2 * np.pi, N)
    return np.stack([center[0] + radius * np.cos(t),
                     center[1] + radius * np.sin(t)], axis=1), t

def infinity(a, b, center, N):
    t = np.linspace(0, 2 * np.pi, N)
    return np.stack([center[0] + b * np.sin(2 * t),
                     center[1] + a * np.sin(t)], axis=1), t

# ----- Calcolo parametri -----
N = 500
pos_min = positions_raw.min(dim=0)[0]
pos_max = positions_raw.max(dim=0)[0]
pos_range = pos_max - pos_min
center = ((pos_max + pos_min) / 2).tolist()

radius = 0.45 * min(pos_range).item()
circle_pts, t_circle = circle(radius, center, N)

margin = 0.05
max_a = (pos_range[1].item() / 2) * (1 - margin)
max_b = (pos_range[0].item() / 2) * (1 - margin)
a_inf = min(0.45 * pos_range[0].item(), max_a)
b_inf = min(0.45 * pos_range[1].item(), max_b)
infinity_pts, t_infinity = infinity(a_inf, b_inf, center, N)

circle_pts_tensor = torch.tensor(circle_pts, dtype=torch.float32)
infinity_pts_tensor = torch.tensor(infinity_pts, dtype=torch.float32)

# ----- Range per KAN -----
all_positions = torch.cat([positions_raw], dim=0)
pos_min = all_positions.min(dim=0)[0]
pos_max = all_positions.max(dim=0)[0]
padding_pos = 0.1 * (pos_max - pos_min)
x_min_pos = (pos_min - padding_pos).tolist()
x_max_pos = (pos_max + padding_pos).tolist()

act_min = actuations.min()
act_max = actuations.max()
padding_act = 0.1 * (act_max - act_min)
x_min_act = (act_min - padding_act).item()
x_max_act = (act_max + padding_act).item()

# ----- Dataset -----
split = int(0.8 * len(actuations))
train_fwd = TensorDataset(actuations[:split], positions[:split])
train_inv = TensorDataset(positions[:split], actuations[:split])

loader_fwd = DataLoader(train_fwd, batch_size=64, shuffle=True)
loader_inv = DataLoader(train_inv, batch_size=64, shuffle=True)

# ----- MLP -----
mlp_inv_model = MLP_Net(input_dim=2, hidden_dims=[64, 32], output_dim=9).to(device)
mlp_fwd_model = MLP_Net(input_dim=9, hidden_dims=[64, 32], output_dim=2).to(device)

# ----- KAN -----
inv_configs = [
    {"out_features": 64, "n_knots": 16, "x_min": -1.0, "x_max": 1.0, "use_bn": True,
     "dropout": 0.1},
    {"out_features": 32, "n_knots": 12, "x_min": -1.0, "x_max": 1.0, "use_bn": False,
     "dropout": 0.0},
]
fwd_configs = [
    {"out_features": 64, "n_knots": 16, "x_min": -1.0, "x_max": 1.0, "use_bn": True,
     "dropout": 0.1},
    {"out_features": 32, "n_knots": 12, "x_min": -1.0, "x_max": 1.0, "use_bn": False,
     "dropout": 0.0},
]
kan_inv_model = KAN_Net(2, inv_configs, 9).to(device)
kan_fwd_model = KAN_Net(9, fwd_configs, 2).to(device)

# ----- Training loop -----
def train_models(inv_model, fwd_model, name):
    criterion = nn.MSELoss()
    opt_inv = optim.Adam(inv_model.parameters(), lr=5e-4, weight_decay=1e-5)
    opt_fwd = optim.Adam(fwd_model.parameters(), lr=5e-4, weight_decay=1e-5)

    for epoch in range(1, 51):
        inv_model.train()
        fwd_model.train()
        tot_loss = 0.0
        for (pos_in, act_gt), (act_in, pos_gt) in zip(loader_inv, loader_fwd):
            pos_in, act_gt = pos_in.to(device), act_gt.to(device)
            act_in, pos_gt = act_in.to(device), pos_gt.to(device)

            opt_fwd.zero_grad()
            pos_pred = fwd_model(act_in)
            loss_fwd = criterion(pos_pred, pos_gt)

            opt_inv.zero_grad()
            act_pred = inv_model(pos_in)
            loss_inv = criterion(act_pred, act_gt)

            pos_cycle = fwd_model(act_pred)
            loss_cycle = criterion(pos_cycle, pos_in)

            loss = loss_fwd + loss_inv + 0.5 * loss_cycle
            loss.backward()
            opt_fwd.step()
            opt_inv.step()

            tot_loss += loss.item()
        print(f"[{name}] Epoch {epoch:02d} | Total Loss: {tot_loss:.4f}")

    torch.save(inv_model.state_dict(), f"{name.lower()}_inv_model.pth")
    torch.save(fwd_model.state_dict(), f"{name.lower()}_fwd_model.pth")

    return inv_model, fwd_model

# ----- Run training -----
train_models(mlp_inv_model, mlp_fwd_model, "MLP")
train_models(kan_inv_model, kan_fwd_model, "KAN")

# ----- Evaluation and Plotting -----
def evaluate_models(mlp_inv, mlp_fwd, kan_inv, kan_fwd):
    mlp_inv.eval()
    mlp_fwd.eval()
    kan_inv.eval()
    kan_fwd.eval()
    with torch.no_grad():
        c_norm = (circle_pts_tensor - mean_pos) / std_pos
        i_norm = (infinity_pts_tensor - mean_pos) / std_pos

        circle_act_mlp = mlp_inv(c_norm.to(device))
        inf_act_mlp = mlp_inv(i_norm.to(device))
        circle_rec_mlp = mlp_fwd(circle_act_mlp).cpu()
        inf_rec_mlp = mlp_fwd(inf_act_mlp).cpu()

        circle_act_kan = kan_inv(c_norm.to(device))
        inf_act_kan = kan_inv(i_norm.to(device))
        circle_rec_kan = kan_fwd(circle_act_kan).cpu()
        inf_rec_kan = kan_fwd(inf_act_kan).cpu()

    circle_rec_mlp = denormalize(circle_rec_mlp, mean_pos, std_pos).numpy()
    inf_rec_mlp = denormalize(inf_rec_mlp, mean_pos, std_pos).numpy()
    circle_rec_kan = denormalize(circle_rec_kan, mean_pos, std_pos).numpy()
    inf_rec_kan = denormalize(inf_rec_kan, mean_pos, std_pos).numpy()

    circle_rec_mlp += (circle_pts.mean(axis=0) - circle_rec_mlp.mean(axis=0))
    inf_rec_mlp += (infinity_pts.mean(axis=0) - inf_rec_mlp.mean(axis=0))
    circle_rec_kan += (circle_pts.mean(axis=0) - circle_rec_kan.mean(axis=0))
    inf_rec_kan += (infinity_pts.mean(axis=0) - inf_rec_kan.mean(axis=0))

    print(f"🔵 MSE Cerchio (MLP):  {mean_squared_error(circle_pts, circle_rec_mlp):.6f}")
    print(f"🔁 MSE Infinito (MLP): {mean_squared_error(infinity_pts, inf_rec_mlp):.6f}")
    print(f"🔵 MSE Cerchio (KAN):  {mean_squared_error(circle_pts, circle_rec_kan):.6f}")
    print(f"🔁 MSE Infinito (KAN): {mean_squared_error(infinity_pts, inf_rec_kan):.6f}")

    os.makedirs("plots/plot_inv", exist_ok=True)

    plot_colored_trajectory_comp(
        circle_pts, circle_rec_kan, t_circle, "Cerchio",
        "plots/plot_inv/circle_compare_kan_mlp.png", "KAN",
        rec_pts_mlp=circle_rec_mlp, name_mlp="MLP", workspace_pts=positions_raw.numpy()
    )

    plot_colored_trajectory_comp(
        infinity_pts, inf_rec_kan, t_infinity, "Infinito",
        "plots/plot_inv/infinity_compare_kan_mlp.png", "KAN",
        rec_pts_mlp=inf_rec_mlp, name_mlp="MLP", workspace_pts=positions_raw.numpy()
    )

    plot_actuations_comp(
        t_circle, circle_act_kan[:, :3], "Attuazioni - Cerchio (KAN vs MLP)",
        "plots/plot_inv/actuation_circle_compare_kan_mlp.png", actuations_mlp=circle_act_mlp[:, :3]
    )

    plot_actuations_comp(
        t_infinity, inf_act_kan[:, :3], "Attuazioni - Infinito (KAN vs MLP)",
        "plots/plot_inv/actuation_infinity_compare_kan_mlp.png", actuations_mlp=inf_act_mlp[:, :3]
    )

# ----- Run evaluation -----
evaluate_models(mlp_inv_model, mlp_fwd_model, kan_inv_model, kan_fwd_model)