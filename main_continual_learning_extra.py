import os
import datetime
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset.data_loader import DataLoader as MyDataLoader
from src.KAN.KAN_Net import KAN_Net
from src.MLP.MLP_Net import MLP_Net
from src.utility import (
    train_one_epoch,
    evaluate_and_save,
    evaluate_model,
    plot_model_vs_itself,
    plot_workspace_split_from_splitdata,
    visualize_performance
)

# -------------------------------
# 1) Hyperparametri e device
# -------------------------------
device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size     = 64
epochs_phase1  = 80    # MsDs
epochs_phase2  = 60    # MdDd
max_grad_norm  = 1.0
results_dir    = "plots/results"
os.makedirs(results_dir, exist_ok=True)

# -------------------------------
# 2) Funzione di calcolo test‐loss
# -------------------------------
def compute_test_loss(model, loader, device, loss_fn):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total += loss_fn(pred, y).item() * x.size(0)
            count += x.size(0)
    return total / count

loss_fn = nn.MSELoss()

# -------------------------------
# 3) Caricamento e normalizzazione dati
# -------------------------------
dl_ds = MyDataLoader(); dl_ds.load_data(deformation="bending",    trial_num=2)
dl_dd = MyDataLoader(); dl_dd.load_data(deformation="twisting_CW", trial_num=2)

# raw tensors
X_raw_ds = torch.tensor(dl_ds.get_data()['actuation'], dtype=torch.float32)
Y_ds     = torch.tensor(dl_ds.get_data()['markers'][:, -1, :], dtype=torch.float32)
X_raw_dd = torch.tensor(dl_dd.get_data()['actuation'], dtype=torch.float32)
Y_dd     = torch.tensor(dl_dd.get_data()['markers'][:, -1, :], dtype=torch.float32)

# Normalize X on Ds → [-1,1]
X_min, X_max = X_raw_ds.min(0)[0], X_raw_ds.max(0)[0]
X_ds = 2*(X_raw_ds - X_min)/(X_max - X_min) - 1.0
X_dd = 2*(X_raw_dd - X_min)/(X_max - X_min) - 1.0

# Center & scale Y on Ds+Dd
Y_all    = torch.cat([Y_ds, Y_dd], dim=0)
y_center = Y_all.mean(0)
y_std    = Y_all.std(0) + 1e-8
print("STD Y per asse:", y_std.tolist())
Y_ds_norm = (Y_ds - y_center)/y_std
Y_dd_norm = (Y_dd - y_center)/y_std

# Plot workspace split
os.makedirs(f"{results_dir}/workspace", exist_ok=True)
plot_workspace_split_from_splitdata(Y_ds_norm, Y_dd_norm)

# DataLoader
loader_ds      = DataLoader(TensorDataset(X_ds,     Y_ds_norm), batch_size, shuffle=True)
loader_dd      = DataLoader(TensorDataset(X_dd,     Y_dd_norm), batch_size, shuffle=True)
mem_loader     = DataLoader(TensorDataset(X_ds,     Y_ds_norm), batch_size, shuffle=True)
test_loader_ds = DataLoader(TensorDataset(X_ds,     Y_ds_norm), batch_size, shuffle=False)
test_loader_dd = DataLoader(TensorDataset(X_dd,     Y_dd_norm), batch_size, shuffle=False)

# -------------------------------
# 4) Costruzione modelli + ottimizzatori + scheduler
# -------------------------------
kan = KAN_Net(
    input_dim=9,
    layer_configs=[
        {"out_features": 128, "n_knots": 32, "x_min": -1.0, "x_max": 1.0, "use_bn": True,  "dropout": 0.0},
        {"out_features": 64,  "n_knots": 24, "x_min": -1.0, "x_max": 1.0, "use_bn": False, "dropout": 0.0},
    ],
    output_dim=3
).to(device)

mlp = MLP_Net(input_dim=9, hidden_dims=[64, 32], output_dim=3).to(device)

opt_kan = torch.optim.AdamW(kan.parameters(), lr=1e-3, weight_decay=1e-4)
opt_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-3)

sched_kan = ReduceLROnPlateau(opt_kan, mode='min', factor=0.5, patience=5, verbose=True)
sched_mlp = ReduceLROnPlateau(opt_mlp, mode='min', factor=0.5, patience=5, verbose=True)

# -------------------------------
# 5) Phase 1: MsDs (train on Ds=bending)
# -------------------------------
print("=== Phase 1: MsDs (bending) ===")
test_losses_kan_MsDs = []
test_losses_mlp_MsDs = []

best_tl_kan = float('inf')
patience, no_improve = 10, 0

for ep in range(1, epochs_phase1+1):
    l_kan = train_one_epoch(kan, loader_ds, loss_fn, opt_kan, device)
    l_mlp = train_one_epoch(mlp, loader_ds, loss_fn, opt_mlp, device)
    torch.nn.utils.clip_grad_norm_(kan.parameters(), max_grad_norm)
    torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_grad_norm)

    tl_kan = compute_test_loss(kan, test_loader_ds, device, loss_fn)
    tl_mlp = compute_test_loss(mlp, test_loader_ds, device, loss_fn)
    test_losses_kan_MsDs.append(tl_kan)
    test_losses_mlp_MsDs.append(tl_mlp)

    sched_kan.step(tl_kan)
    sched_mlp.step(tl_mlp)

    print(f"[MsDs] Ep {ep}/{epochs_phase1} | "
          f"Train KAN {l_kan:.4f}, MLP {l_mlp:.4f} | "
          f"Test KAN {tl_kan:.4f}, MLP {tl_mlp:.4f}")

    # Early stopping KAN su MsDs
    if tl_kan < best_tl_kan:
        best_tl_kan, no_improve = tl_kan, 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"→ Early stopping KAN a MsDs dopo {ep} epoche")
            break

# Visualizzo performance su Ds
visualize_performance(
    test_losses_kan_MsDs,
    test_losses_mlp_MsDs,
    kan, mlp,
    test_loader_ds,
    device,
    name="MsDs"
)

# Salvo valutazioni finali MsDs
evaluate_and_save(
    {'KAN': kan, 'MLP': mlp},
    X_ds, Y_ds_norm,
    device, evaluate_model, results_dir,
    label='MsDs'
)

# -------------------------------
# 6) Phase 2: MdDd (fine-tune su Dd=twisting) con rehearsal
# -------------------------------
print("=== Phase 2: MdDd (twisting) w/ rehearsal ===")
test_losses_kan_MdDs = []
test_losses_mlp_MdDs = []
test_losses_kan_MdDd = []
test_losses_mlp_MdDd = []

for ep in range(1, epochs_phase2+1):
    iter_mem = iter(mem_loader)
    for idx, (x_dd, y_dd) in enumerate(loader_dd):
        # batch di memoria ogni 2 batch di twisting
        x_mem, y_mem = next(iter_mem, (None, None))
        if x_mem is not None and idx % 2 == 0:
            x_batch = torch.cat([x_dd, x_mem], dim=0).to(device)
            y_batch = torch.cat([y_dd, y_mem], dim=0).to(device)
        else:
            x_batch, y_batch = x_dd.to(device), y_dd.to(device)

        # KAN update
        loss_k = loss_fn(kan(x_batch), y_batch)
        opt_kan.zero_grad(); loss_k.backward()
        torch.nn.utils.clip_grad_norm_(kan.parameters(), max_grad_norm)
        opt_kan.step()

        # MLP update
        loss_m = loss_fn(mlp(x_batch), y_batch)
        opt_mlp.zero_grad(); loss_m.backward()
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_grad_norm)
        opt_mlp.step()

    # valutazione su Ds e su Dd
    tl_k_ds = compute_test_loss(kan, test_loader_ds, device, loss_fn)
    tl_m_ds = compute_test_loss(mlp, test_loader_ds, device, loss_fn)
    tl_k_dd = compute_test_loss(kan, test_loader_dd, device, loss_fn)
    tl_m_dd = compute_test_loss(mlp, test_loader_dd, device, loss_fn)

    test_losses_kan_MdDs.append(tl_k_ds)
    test_losses_mlp_MdDs.append(tl_m_ds)
    test_losses_kan_MdDd.append(tl_k_dd)
    test_losses_mlp_MdDd.append(tl_m_dd)

    print(f"[MdDd] Ep {ep}/{epochs_phase2} | "
          f"KAN@Ds {tl_k_ds:.4f}, MLP@Ds {tl_m_ds:.4f} | "
          f"KAN@Dd {tl_k_dd:.4f}, MLP@Dd {tl_m_dd:.4f}")

# Visualizzo performance MdDs (su Ds) e MdDd (su Dd)
visualize_performance(test_losses_kan_MdDs, test_losses_mlp_MdDs,
                      kan, mlp, test_loader_ds, device, name="MdDs")
visualize_performance(test_losses_kan_MdDd, test_losses_mlp_MdDd,
                      kan, mlp, test_loader_dd, device, name="MdDd")

# Salvo valutazioni finali
evaluate_and_save(
    {'KAN': kan, 'MLP': mlp},
    X_ds, Y_ds_norm,
    device, evaluate_model, results_dir,
    label='MdDs'
)
evaluate_and_save(
    {'KAN': kan, 'MLP': mlp},
    X_dd, Y_dd_norm,
    device, evaluate_model, results_dir,
    label='MdDd'
)

# -------------------------------
# 7) Plots comparativi MsDs vs MdDs
# -------------------------------
for model_name in ['KAN', 'MLP']:
    for metric in ['X_RMSE','Y_RMSE','Z_RMSE','X_R2','Y_R2','Z_R2']:
        plot_model_vs_itself(
            results_dir,
            model_name=model_name,
            metric=metric,
            labels=('MsDs','MdDs')
        )
