import torch
import numpy as np
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import DataLoader, TensorDataset
from dataset.data_loader import DataLoader as MyDataLoader
from sklearn.metrics import mean_squared_error, r2_score  # aggiungi questo in cima al file
from dataset.data_loader import DataLoader as MyLoader


def prepare_sequence_loaders(deformation, train_trial, test_trial, seq_len, batch_size):
    def load_and_process(trial):
        loader = MyLoader()
        loader.load_data(deformation=deformation, trial_num=trial)
        data = loader.get_data()
        X = data["actuation"]
        Y = data["markers"][:, -1, :]  # ultimo marker
        n_samples = X.shape[0] - seq_len
        X_seq = np.array([X[i:i + seq_len] for i in range(n_samples)])
        Y_seq = np.array([Y[i + seq_len] for i in range(n_samples)])
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(Y_seq, dtype=torch.float32)

    X_train, Y_train = load_and_process(train_trial)
    X_test, Y_test = load_and_process(test_trial)

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False)

    in_dim = X_train.shape[-1]
    out_dim = Y_train.shape[-1]
    return train_loader, test_loader, in_dim, out_dim


def prepare_loaders(
        deformation: str = "bending",
        train_trial: int = 1,
        test_trial: int = 2,
        batch_size: int = 64
):
    # Train data
    dl = MyDataLoader()
    dl.load_data(deformation=deformation, trial_num=train_trial)
    data_tr = dl.get_data()
    X_tr = torch.tensor(data_tr["actuation"], dtype=torch.float32)
    Y_tr = torch.tensor(data_tr["markers"][:, -1, :], dtype=torch.float32)

    # Test data
    dl_2 = MyDataLoader()
    dl_2.load_data(deformation=deformation, trial_num=test_trial)
    data_ts = dl_2.get_data()
    X_ts = torch.tensor(data_ts["actuation"], dtype=torch.float32)
    Y_ts = torch.tensor(data_ts["markers"][:, -1, :], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_ts, Y_ts), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_one_epoch(model, loader, loss_fn, opt, device):
    model.train()
    total = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        opt.step()
        total += loss.item() * x.size(0)

    return total / len(loader.dataset)


def eval_loss(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += loss_fn(model(x), y).item() * x.size(0)

    return total / len(loader.dataset)


def visualize_performance(test_losses_kan, test_losses_mlp, kan, mlp, test_loader, device):
    """
    Plotta:
      1) Test loss vs epoca per KAN e MLP
      2) Scatter true vs pred (X, Y, Z) sul test set
      3) RMSE e R² per ogni coordinata
      4) Salva tutti i plot nella cartella plot/YYYY-MM-DD/
    """

    # Crea cartella plot/data
    today = datetime.date.today().isoformat()
    plot_dir = os.path.join("plot", today)
    os.makedirs(plot_dir, exist_ok=True)

    # 1) Curve di test loss
    epochs = range(1, len(test_losses_kan) + 1)
    plt.figure()
    plt.plot(epochs, test_losses_kan, label='KAN Test Loss')
    plt.plot(epochs, test_losses_mlp, label='MLP Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Test Loss su Bending Trial')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'test_loss_curve.png'))
    plt.close()

    # 2) Raccolta predizioni
    kan.eval()
    mlp.eval()
    y_true, y_pred_kan, y_pred_mlp = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            y_true.append(y.cpu().numpy())
            y_pred_kan.append(kan(x).cpu().numpy())
            y_pred_mlp.append(mlp(x).cpu().numpy())

    y_true = np.vstack(y_true)
    y_pred_kan = np.vstack(y_pred_kan)
    y_pred_mlp = np.vstack(y_pred_mlp)

    # 3) Scatter plot e metriche
    coords = ['X', 'Y', 'Z']
    print("\n Metriche di performance sul test set (Bending Trial):")
    for i, coord in enumerate(coords):
        plt.figure()
        plt.scatter(y_true[:, i], y_pred_kan[:, i], alpha=0.3, label='KAN')
        plt.scatter(y_true[:, i], y_pred_mlp[:, i], alpha=0.3, label='MLP')
        mn, mx = y_true[:, i].min(), y_true[:, i].max()
        plt.plot([mn, mx], [mn, mx], 'k--', linewidth=1)
        plt.xlabel(f'True {coord} position')
        plt.ylabel(f'Predicted {coord} position')
        plt.title(f'True vs Predicted ({coord}) on Bending Test')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f'true_vs_pred_{coord}.png'))
        plt.close()

        # Metriche (RMSE calcolato manualmente per compatibilità)
        rmse_kan = np.sqrt(mean_squared_error(y_true[:, i], y_pred_kan[:, i]))
        rmse_mlp = np.sqrt(mean_squared_error(y_true[:, i], y_pred_mlp[:, i]))
        r2_kan = r2_score(y_true[:, i], y_pred_kan[:, i])
        r2_mlp = r2_score(y_true[:, i], y_pred_mlp[:, i])

        print(f"{coord}:")
        print(f"  KAN ➜ RMSE = {rmse_kan:.4f}, R² = {r2_kan:.4f}")
        print(f"  MLP ➜ RMSE = {rmse_mlp:.4f}, R² = {r2_mlp:.4f}")


def visualize_performance_rnn(
        test_losses_kan: list,
        test_losses_mlp: list,
        test_losses_rnn: list,
        kan_model: torch.nn.Module,
        mlp_model: torch.nn.Module,
        rnn_model: torch.nn.Module,
        static_loader: torch.utils.data.DataLoader,
        seq_loader: torch.utils.data.DataLoader,
        device: torch.device
):
    """
    Plotta KAN, MLP e RNN:
      1) Curva di test loss vs epoca per tutti e tre
      2) Scatter true vs pred (X,Y,Z) con tre serie (KAN, MLP, RNN)
      3) Stampare RMSE e R² per ciascun modello e coordinata
      4) Salva tutti i plot in plot/YYYY-MM-DD/all_models/
    """
    # 0) Prepare output dir
    today = datetime.date.today().isoformat()
    plot_dir = os.path.join("plot", today, "all_models")
    os.makedirs(plot_dir, exist_ok=True)

    # 1) Loss curves
    plt.figure()
    plt.plot(range(1, len(test_losses_kan) + 1), test_losses_kan, marker='o', label='KAN')
    plt.plot(range(1, len(test_losses_mlp) + 1), test_losses_mlp, marker='s', label='MLP')
    plt.plot(range(1, len(test_losses_rnn) + 1), test_losses_rnn, marker='^', label='RNN')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Test Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'loss_curves_all.png'))
    plt.close()

    # 2) Collect predictions for static (KAN & MLP)
    kan_model.eval()
    mlp_model.eval()
    rnn_model.eval()
    y_true_stat, y_pred_kan, y_pred_mlp = [], [], []
    with torch.no_grad():
        for x, y in static_loader:
            x = x.to(device)
            y = y.to(device)
            y_true_stat.append(y.cpu().numpy())
            y_pred_kan.append(kan_model(x).cpu().numpy())
            y_pred_mlp.append(mlp_model(x).cpu().numpy())
    y_true_stat = np.vstack(y_true_stat)
    y_pred_kan = np.vstack(y_pred_kan)
    y_pred_mlp = np.vstack(y_pred_mlp)

    # 3) Collect predictions for rnn (dynamic)
    y_true_dyn, y_pred_rnn = [], []
    with torch.no_grad():
        for x_seq, y_seq in seq_loader:
            x_seq = x_seq.to(device)
            y_seq = y_seq.to(device)
            y_true_dyn.append(y_seq.cpu().numpy())
            y_pred_rnn.append(rnn_model(x_seq).cpu().numpy())
    y_true_dyn = np.vstack(y_true_dyn)
    y_pred_rnn = np.vstack(y_pred_rnn)

    # 4) Scatter + metrics per coordinata
    coords = ['X', 'Y', 'Z']
    print("\n=== Performance Summary ===")
    for i, c in enumerate(coords):
        plt.figure()
        # scatter
        plt.scatter(y_true_stat[:, i], y_pred_kan[:, i], alpha=0.3, label='KAN')
        plt.scatter(y_true_stat[:, i], y_pred_mlp[:, i], alpha=0.3, label='MLP')
        plt.scatter(y_true_dyn[:, i], y_pred_rnn[:, i], alpha=0.3, label='RNN')
        # diagonal
        mn = min(y_true_stat[:, i].min(), y_true_dyn[:, i].min())
        mx = max(y_true_stat[:, i].max(), y_true_dyn[:, i].max())
        plt.plot([mn, mx], [mn, mx], 'k--', linewidth=1)
        plt.xlabel(f'True {c}')
        plt.ylabel(f'Pred {c}')
        plt.title(f'True vs Pred ({c})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f'scatter_{c}.png'))
        plt.close()

        # metriche
        rmse_kan = np.sqrt(mean_squared_error(y_true_stat[:, i], y_pred_kan[:, i]))
        r2_kan = r2_score(y_true_stat[:, i], y_pred_kan[:, i])
        rmse_mlp = np.sqrt(mean_squared_error(y_true_stat[:, i], y_pred_mlp[:, i]))
        r2_mlp = r2_score(y_true_stat[:, i], y_pred_mlp[:, i])
        rmse_rnn = np.sqrt(mean_squared_error(y_true_dyn[:, i], y_pred_rnn[:, i]))
        r2_rnn = r2_score(y_true_dyn[:, i], y_pred_rnn[:, i])

        print(f"{c}:")
        print(f"  KAN ➜ RMSE={rmse_kan:.4f}, R²={r2_kan:.4f}")
        print(f"  MLP ➜ RMSE={rmse_mlp:.4f}, R²={r2_mlp:.4f}")
        print(f"  RNN ➜ RMSE={rmse_rnn:.4f}, R²={r2_rnn:.4f}")

    # Done
    print(f"\nPlots saved to {plot_dir}")


def split_dataset_by_tip_position(X, Y, axis=1, threshold=0.0, twisting=False):
    """
    Split the dataset into LEFT/RIGHT semispaces based on tip position.

    If twisting is False, split along the specified Y[:, axis] vs threshold.
    If twisting is True, split along the XY diagonal (Y[:,0] - Y[:,1] >= 0).
    """
    if twisting:
        tip_coord = Y[:, 0] - Y[:, 1]  # Diagonal: X - Y
        threshold = 0.0
        print("Using diagonal (X - Y) for twisting split.")
    else:
        tip_coord = Y[:, axis]
        print(f"Split on axis {axis} at {threshold:.4f}")

    mask_right = tip_coord >= threshold
    mask_left = tip_coord < threshold

    print(f"Right: {mask_right.sum().item()}, Left: {mask_left.sum().item()}")
    return {
        'right': (X[mask_right], Y[mask_right]),
        'left': (X[mask_left], Y[mask_left])
    }


def plot_workspace_split_from_splitdata(Y_left, Y_right, title='Workspace split view (Top-down)',
                                        filename='workspace_split.png'):
    """
    Plot workspace (X vs Y) for left and right datasets separately and save the figure.

    Args:
        Y_left: numpy array of shape (N, 3) with tip positions classified as 'left'.
        Y_right: numpy array of shape (M, 3) with tip positions classified as 'right'.
        title: Title of the plot.
        filename: Name of the file to save the plot (e.g. 'workspace_split.png').
    """
    Y_left = Y_left.reshape(-1, 3)
    Y_right = Y_right.reshape(-1, 3)

    x_left, y_left = Y_left[:, 0], Y_left[:, 1]
    x_right, y_right = Y_right[:, 0], Y_right[:, 1]

    plt.figure(figsize=(6, 6))
    plt.scatter(x_right, y_right, s=10, color='orange', label='Right', alpha=0.6)
    plt.scatter(x_left, y_left, s=10, color='blue', label='Left', alpha=0.6)
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title(title)
    plt.legend()
    plt.axis('equal')
    plt.grid(True)

    # Salvataggio
    output_dir = os.path.join("plots", "results", "workspace")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Workspace plot saved to: {save_path}")


def compute_metrics(y_true, y_pred):
    """
    Calcola RMSE e R² per ogni colonna (X, Y, Z)
    """
    metrics = {}
    coords = ['X', 'Y', 'Z']
    for i, coord in enumerate(coords):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        metrics[coord] = {'RMSE': rmse, 'R2': r2}
    return metrics


def evaluate_model(model, X, Y, device):
    model.eval()
    with torch.no_grad():
        X, Y = X.to(device), Y.to(device)
        pred = model(X).cpu().numpy()
        true = Y.cpu().numpy()
    return {f"{k}_{m}": v for k, v in compute_metrics(true, pred).items() for m, v in v.items()}


def plot_model_vs_itself(results_dir, model_name, metric='X_RMSE', labels=('MsDs', 'MdDs')):
    values = []
    for label in labels:
        csv_path = os.path.join(results_dir, f"evaluation_{label}.csv")
        df = pd.read_csv(csv_path)
        row = df[df['model'] == model_name].iloc[0]
        values.append(row[metric])

    plt.figure()
    plt.bar(labels, values)
    plt.title(f'{model_name} – {metric} comparison')
    plt.ylabel(metric)
    plt.xlabel('Scenario')
    plt.savefig(os.path.join(results_dir, f'{model_name}_{metric}_comparison.png'))
    plt.close()


def evaluate_and_save(models, X, Y, device, evaluate_model_fn, results_dir, label):
    """
    Valuta i modelli, stampa metriche e salva risultati (CSV e NPZ).

    Args:
        models (dict): mappa nome->modello
        X (Tensor): dati di input
        Y (Tensor): dati target
        device (torch.device)
        evaluate_model_fn (callable): funzione che ritorna metrics dict
        results_dir (str): cartella di salvataggio
        label (str): nome della fase (es. MsDs)
    """
    os.makedirs(results_dir, exist_ok=True)
    preds = {'true': Y.cpu().numpy()}
    records = []
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            y_pred = model(X.to(device)).cpu().numpy()
            preds[f'pred_{name}'] = y_pred
            metrics = evaluate_model_fn(model, X, Y, device)
            records.append({'side': label, 'model': name, **metrics})
            print(f"[{label}] {name}: {metrics}")
    # salva
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(results_dir, f'evaluation_{label}.csv'), index=False)
    np.savez(os.path.join(results_dir, f'predictions_{label}.npz'), **preds)


def extract_equations_from_kan(model, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)

    layer0 = model.blocks[0].kan  # Primo KAN_Layer
    x_knots, y_coeffs = layer0.export_splines()  # y_coeffs: [out_dim, in_dim, n_knots]

    out_dim, in_dim, _ = y_coeffs.shape

    # Prendiamo solo primi 3 attuatori e primi 3 output
    selected_inputs = [0, 1, 2]  # u1, u2, u3
    selected_outputs = [0, 1, 2]  # x, y, z

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 8), sharex=True, sharey=True)

    for i, out_idx in enumerate(selected_outputs):
        for j, in_idx in enumerate(selected_inputs):
            x = x_knots
            y = y_coeffs[out_idx, in_idx]
            ax = axes[i, j]
            ax.plot(x, y, marker='o')
            ax.set_title(f'$f_{{{in_idx + 1}}}(u_{in_idx + 1}) \\to {["x", "y", "z"][i]}$')
            ax.grid(True)

    fig.suptitle("Spline Functions Mapping Actuators to Coordinates", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    plot_path = os.path.join(plot_dir, "kan_splines_grid.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Spline plot salvato in: {plot_path}")


def plot_colored_trajectory_comp(gt_pts, rec_pts_kan, t, title, filename, name_kan,
                                 rec_pts_mlp=None, name_mlp="MLP", workspace_pts=None):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Inversione assi
    x_gt, y_gt = gt_pts[:, 1], gt_pts[:, 0]
    x_rec_kan, y_rec_kan = rec_pts_kan[:, 1], rec_pts_kan[:, 0]

    # Workspace
    if workspace_pts is not None:
        x_ws, y_ws = workspace_pts[:, 1], workspace_pts[:, 0]
        x_center = (x_ws.min() + x_ws.max()) / 2
        left_mask = x_ws < x_center
        right_mask = x_ws >= x_center
        ax.scatter(x_ws[left_mask], y_ws[left_mask], color='lightskyblue', s=5, alpha=0.5, label='Workspace (sx)')
        ax.scatter(x_ws[right_mask], y_ws[right_mask], color='dodgerblue', s=5, alpha=0.5, label='Workspace (dx)')

    # Target: linea tratteggiata nera
    ax.plot(x_gt, y_gt, 'k--', label='Target', linewidth=2)

    # Predizione KAN (rossa)
    scatter_kan = ax.scatter(x_rec_kan, y_rec_kan, c=t, cmap='autumn', alpha=0.95, s=30, label=f'Predicted {name_kan}')

    # Predizione MLP (verde) — aggiungiamo il secondo scatter
    scatter_mlp = None
    if rec_pts_mlp is not None:
        x_rec_mlp, y_rec_mlp = rec_pts_mlp[:, 1], rec_pts_mlp[:, 0]
        scatter_mlp = ax.scatter(x_rec_mlp, y_rec_mlp, c=t, cmap='summer', alpha=0.9, s=20, marker='x',
                                 label=f'Predicted {name_mlp}')

    ax.set_xlabel("y")
    ax.set_ylabel("x")
    ax.set_title(f"Traiettoria nel workspace - {title}")
    ax.axis('equal')
    ax.grid(True)

    # Divider per inserire due colorbar
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="4%", pad=0.1)
    cb1 = plt.colorbar(scatter_kan, cax=cax1)
    cb1.set_label("Tempo (KAN)")

    if scatter_mlp is not None:
        cax2 = divider.append_axes("right", size="4%", pad=0.6)
        norm_mlp = mcolors.Normalize(vmin=min(t), vmax=max(t))
        cb2 = plt.colorbar(cm.ScalarMappable(norm=norm_mlp, cmap='summer'), cax=cax2)
        cb2.set_label("Tempo (MLP)")

    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_actuations_comp(t, actuations_kan, title, filename, actuations_mlp=None, label_kan="KAN", label_mlp="MLP"):
    plt.figure(figsize=(6, 4))
    colors = cm.tab10.colors  # 10 colori distinti

    # Plot attuazioni KAN (colori pieni)
    for k in range(actuations_kan.shape[1]):
        plt.plot(t, actuations_kan[:, k].cpu(), label=f'{label_kan} - Attuatore {k + 1}', color=colors[k % 10])

    # Plot attuazioni MLP (stile tratteggiato)
    if actuations_mlp is not None:
        for k in range(actuations_mlp.shape[1]):
            plt.plot(t, actuations_mlp[:, k].cpu(), linestyle='--', label=f'{label_mlp} - Attuatore {k + 1}',
                     color=colors[k % 10])

    plt.title(title)
    plt.xlabel("Tempo")
    plt.ylabel("Lunghezza")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_rnn_predictions_workspace(model, loader, device, axis=0, title_prefix="Model"):
    """
       Plots the absolute error along a given axis projected in the XY workspace.

       Parameters:
       - model: trained model
       - loader: test dataloader (static or sequential)
       - device: 'cuda' or 'cpu'
       - axis: 0 for X, 1 for Y, 2 for Z
       - title_prefix: string to prepend in the plot title
       """
    model.eval()
    preds_all, targets_all = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu()
            preds_all.append(preds)
            targets_all.append(y_batch)

    preds_all = torch.cat(preds_all).numpy()
    targets_all = torch.cat(targets_all).numpy()

    abs_error = np.abs(preds_all[:, axis] - targets_all[:, axis])
    x_coords = targets_all[:, 0]
    y_coords = targets_all[:, 1]

    plt.figure(figsize=(8, 7))
    scatter = plt.scatter(x_coords, y_coords, c=abs_error, cmap='viridis', s=15, alpha=0.9)
    plt.colorbar(scatter, label=f"{['X', 'Y', 'Z'][axis]}-axis absolute error")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title(f"{title_prefix} – {['X', 'Y', 'Z'][axis]}-Axis Workspace Error")
    plt.grid(True)
    plt.tight_layout()
    plt.show()