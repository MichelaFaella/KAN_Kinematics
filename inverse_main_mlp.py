import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from src.MLP.MLP_Net import MLP_Net

# ------------------------------
# Device setup
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using", device)

# ------------------------------
# Ground truth forward model
# ------------------------------
def direct_kinematics(cables: torch.Tensor) -> torch.Tensor:
    W = torch.tensor([[0.2, -0.1, 0.1],
                      [0.1,  0.3, -0.2]], dtype=torch.float32)
    b = torch.tensor([0.0, 0.0], dtype=torch.float32)
    return cables @ W.T + b

# ------------------------------
# Dataset creation
# ------------------------------
n_samples = 100000
cables = torch.rand(n_samples, 3)
positions = direct_kinematics(cables)

split = int(0.8 * n_samples)
train_fwd = TensorDataset(cables[:split], positions[:split])
val_fwd = TensorDataset(cables[split:], positions[split:])
train_inv = TensorDataset(positions[:split], cables[:split])
val_inv = TensorDataset(positions[split:], cables[split:])

loader_fwd = DataLoader(train_fwd, batch_size=128, shuffle=True)
loader_inv = DataLoader(train_inv, batch_size=128, shuffle=True)

# ------------------------------
# Model setup
# ------------------------------
inv_model = MLP_Net(input_dim=2, hidden_dims=[64, 32], output_dim=3).to(device)
fwd_model = MLP_Net(input_dim=3, hidden_dims=[64, 32], output_dim=2).to(device)

criterion = nn.MSELoss()
opt_inv = optim.Adam(inv_model.parameters(), lr=5e-4, weight_decay=1e-5)
opt_fwd = optim.Adam(fwd_model.parameters(), lr=5e-4, weight_decay=1e-5)

# ------------------------------
# Training loop
# ------------------------------
epochs = 50
for epoch in range(1, epochs + 1):
    inv_model.train(); fwd_model.train()
    tot_loss = 0.0

    for (p_in, c_gt), (c_in, p_gt) in zip(loader_inv, loader_fwd):
        p_in, c_gt = p_in.to(device), c_gt.to(device)
        c_in, p_gt = c_in.to(device), p_gt.to(device)

        # Forward model training
        opt_fwd.zero_grad()
        p_pred = fwd_model(c_in)
        loss_fwd = criterion(p_pred, p_gt)

        # Inverse model training
        opt_inv.zero_grad()
        c_pred = inv_model(p_in)
        loss_inv = criterion(c_pred, c_gt)

        # Cycle consistency
        p_cycle = fwd_model(c_pred)
        loss_cycle = criterion(p_cycle, p_in)

        # Combined loss
        loss = loss_fwd + loss_inv + 0.5 * loss_cycle
        loss.backward()
        opt_fwd.step()
        opt_inv.step()

        tot_loss += loss.item()

    print(f"Epoch {epoch:02d} | Total Loss: {tot_loss:.4f}")

# ------------------------------
# Save models
# ------------------------------
torch.save(inv_model.state_dict(), "mlp_inv_model.pth")
torch.save(fwd_model.state_dict(), "mlp_fwd_model.pth")

# ------------------------------
# Evaluation: circle + infinity
# ------------------------------
def circle(radius, center, N):
    t = np.linspace(0, 2 * np.pi, N)
    return np.stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)], axis=1), t

def infinity(a, b, center, N):
    t = np.linspace(0, 2 * np.pi, N)
    return np.stack([center[0] + a * np.sin(t), center[1] + b * np.sin(2*t)], axis=1), t

inv_model.eval(); fwd_model.eval()
N = 200
circle_pts, t_circle = circle(0.1, (0, 0), N)
infinity_pts, t_infinity = infinity(0.1, 0.05, (0, 0), N)

with torch.no_grad():
    circle_act = inv_model(torch.from_numpy(circle_pts).float().to(device))
    circle_rec = fwd_model(circle_act).cpu().numpy()

    inf_act = inv_model(torch.from_numpy(infinity_pts).float().to(device))
    inf_rec = fwd_model(inf_act).cpu().numpy()

# ------------------------------
# Plotting
# ------------------------------
plt.figure(figsize=(5, 5))
plt.plot(circle_pts[:, 0], circle_pts[:, 1], 'b--', label='Original')
plt.plot(circle_rec[:, 0], circle_rec[:, 1], 'orange', label='Reconstructed')
plt.title('Circle'); plt.legend(); plt.axis('equal'); plt.grid(True)
plt.savefig("circle_reconstruction.png")

plt.figure(figsize=(5, 5))
plt.plot(infinity_pts[:, 0], infinity_pts[:, 1], 'b--', label='Original')
plt.plot(inf_rec[:, 0], inf_rec[:, 1], 'orange', label='Reconstructed')
plt.title('Infinity'); plt.legend(); plt.axis('equal'); plt.grid(True)
plt.savefig("infinity_reconstruction.png")

plt.figure(figsize=(6, 4))
for k in range(circle_act.shape[1]):
    plt.plot(t_circle, circle_act[:, k].cpu(), label=f'Actuator {k+1}')
plt.title('Actuations - Circle'); plt.legend(); plt.grid(True)
plt.savefig("actuation_circle.png")

plt.figure(figsize=(6, 4))
for k in range(inf_act.shape[1]):
    plt.plot(t_infinity, inf_act[:, k].cpu(), label=f'Actuator {k+1}')
plt.title('Actuations - Infinity'); plt.legend(); plt.grid(True)
plt.savefig("actuation_infinity.png")

plt.show()
