import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from src.KAN.KAN_Net import KAN_Net
from dataset.data_loader import DataLoader as MyDataLoader
from src.utility import plot_colored_trajectory, plot_actuations
import PIL.Image as Image


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

print(f"👉 Shape actuation: {actuations_raw.shape}")
print(f"👉 Shape markers: {positions_raw.shape}")

# ----- Normalizza -----
actuations, mean_act, std_act = normalize(actuations_raw)
positions, mean_pos, std_pos = normalize(positions_raw)

print("✅ Normalizzazione completata")
print(f"Actuation mean: {mean_act.tolist()}")
print(f"Position mean: {mean_pos.tolist()}")


# ----- Traiettorie sintetiche -----
def circle(radius, center, N):
    t = np.linspace(0, 2 * np.pi, N)
    return np.stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)], axis=1), t


def infinity(a, b, center, N):
    t = np.linspace(0, 2 * np.pi, N)
    return np.stack([center[0] + a * np.sin(t), center[1] + b * np.sin(2 * t)], axis=1), t


N = 200
center = mean_pos.tolist()
circle_pts, t_circle = circle(0.1, center, N)
infinity_pts, t_infinity = infinity(0.1, 0.05, center, N)

circle_pts_tensor = torch.tensor(circle_pts, dtype=torch.float32)
infinity_pts_tensor = torch.tensor(infinity_pts, dtype=torch.float32)

# ----- Calcolo range dinamico includendo punti sintetici -----
all_positions = torch.cat([positions, circle_pts_tensor, infinity_pts_tensor], dim=0)
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

print(f"📐 Pos range incl. synthetic: {x_min_pos} → {x_max_pos}")
print(f"📐 Act range padded: {x_min_act:.3f} → {x_max_act:.3f}")

# ----- Split dataset -----
split = int(0.8 * len(actuations))
train_fwd = TensorDataset(actuations[:split], positions[:split])
train_inv = TensorDataset(positions[:split], actuations[:split])

loader_fwd = DataLoader(train_fwd, batch_size=128, shuffle=True)
loader_inv = DataLoader(train_inv, batch_size=128, shuffle=True)

# ----- Modelli -----
inv_configs = [
    {"out_features": 64, "n_knots": 16, "x_min": x_min_pos[0], "x_max": x_max_pos[0], "use_bn": True, "dropout": 0.1},
    {"out_features": 32, "n_knots": 12, "x_min": x_min_pos[1], "x_max": x_max_pos[1], "use_bn": True, "dropout": 0.1},
]
fwd_configs = [
    {"out_features": 64, "n_knots": 16, "x_min": x_min_act, "x_max": x_max_act, "use_bn": True, "dropout": 0.0},
    {"out_features": 32, "n_knots": 12, "x_min": x_min_act, "x_max": x_max_act, "use_bn": True, "dropout": 0.0},
]

inv_model = KAN_Net(2, inv_configs, 9).to(device)
fwd_model = KAN_Net(9, fwd_configs, 2).to(device)

criterion = nn.MSELoss()
opt_inv = optim.Adam(inv_model.parameters(), lr=5e-4, weight_decay=1e-5)
opt_fwd = optim.Adam(fwd_model.parameters(), lr=5e-4, weight_decay=1e-5)

# ----- Training -----
epochs = 50
for epoch in range(1, epochs + 1):
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
    print(f"Epoch {epoch:02d} | Total Loss: {tot_loss:.4f}")

# ----- Salva -----
torch.save(inv_model.state_dict(), "kan_inv_model.pth")
torch.save(fwd_model.state_dict(), "kan_fwd_model.pth")

# ----- Evaluation -----
inv_model.eval()
fwd_model.eval()
with torch.no_grad():
    c_norm = (circle_pts_tensor - mean_pos) / std_pos
    i_norm = (infinity_pts_tensor - mean_pos) / std_pos

    circle_act = inv_model(c_norm.to(device))
    inf_act = inv_model(i_norm.to(device))

    circle_rec = fwd_model(circle_act).cpu()
    inf_rec = fwd_model(inf_act).cpu()

# ----- Denormalizzazione e centratura -----
circle_rec = denormalize(circle_rec, mean_pos, std_pos).numpy()
inf_rec = denormalize(inf_rec, mean_pos, std_pos).numpy()

circle_rec = circle_rec + (circle_pts.mean(axis=0) - circle_rec.mean(axis=0))
inf_rec = inf_rec + (infinity_pts.mean(axis=0) - inf_rec.mean(axis=0))

# ----- MSE -----
print(f"🔵 MSE Cerchio: {mean_squared_error(circle_pts, circle_rec):.6f}")
print(f"🔁 MSE Infinito: {mean_squared_error(infinity_pts, inf_rec):.6f}")

# ----- Plot -----
# ----- Plot -----

os.makedirs("plot_inv", exist_ok=True)

plot_colored_trajectory(circle_pts, circle_rec, t_circle, "Cerchio", "plot_inv/circle_workspace_colorcoded_kan.png", "KAN")
plot_colored_trajectory(infinity_pts, inf_rec, t_infinity, "Infinito", "plot_inv/infinity_workspace_colorcoded_kan.png", "KAN")

plot_actuations(t_circle, circle_act.cpu(), "Attuazioni - Cerchio (KAN)", "plot_inv/actuation_circle_kan.png")
plot_actuations(t_infinity, inf_act.cpu(), "Attuazioni - Infinito (KAN)", "plot_inv/actuation_infinity_kan.png")
