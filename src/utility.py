import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset
from dataset.data_loader import DataLoader as MyDataLoader
from sklearn.metrics import mean_squared_error, r2_score  # aggiungi questo in cima al file


def prepare_sequence_loaders(
        deformation: str,
        train_trial: int,
        test_trial: int,
        seq_len: int,
        batch_size: int = 64
):
    """
    Prepara DataLoader per modelli sequenziali (RNN):
    - X: sequenze di forma [T - seq_len, seq_len, input_dim]
    - Y: target futuro [T - seq_len, output_dim]
    """

    def make_loader(trial_num):
        # carica dati
        dl = MyDataLoader()
        dl.load_data(deformation=deformation, trial_num=trial_num)
        data = dl.get_data()
        X = torch.tensor(data["actuation"], dtype=torch.float32)  # [T, in_dim]
        Y = torch.tensor(data["markers"][:, -1, :], dtype=torch.float32)  # [T, out_dim]

        T, in_dim = X.shape
        _, out_dim = Y.shape
        n_seq = T - seq_len

        # costruisci sequenze
        X_seq = torch.stack([X[i: i + seq_len] for i in range(n_seq)])  # [n_seq, seq_len, in_dim]
        Y_seq = torch.stack([Y[i + seq_len] for i in range(n_seq)])  # [n_seq, out_dim]

        ds = TensorDataset(X_seq, Y_seq)
        return DataLoader(ds, batch_size=batch_size, shuffle=(trial_num == train_trial)), in_dim, out_dim

    train_loader, in_dim, out_dim = make_loader(train_trial)
    test_loader, _, _ = make_loader(test_trial)
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
    kan_model.eval();
    mlp_model.eval();
    rnn_model.eval()
    y_true_stat, y_pred_kan, y_pred_mlp = [], [], []
    with torch.no_grad():
        for x, y in static_loader:
            x = x.to(device);
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
            x_seq = x_seq.to(device);
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


def split_dataset_by_tip_position(X, Y, axis=0, threshold=None):
    """
    Divide il dataset in due semipiani (LEFT e RIGHT) in base alla posizione media
    del tip lungo un asse nel task space (Y).

    Args:
        X (Tensor): attuazioni [N, 9]
        Y (Tensor): posizioni del tip [N, 3]
        axis (int): asse del task space su cui dividere (0=x, 1=y, 2=z)
        threshold (float): valore soglia opzionale per la divisione

    Returns:
        dict: {'left': (X_left, Y_left), 'right': (X_right, Y_right)}
    """
    tip_coord = Y[:, axis]
    if threshold is None:
        threshold = torch.median(tip_coord).item()

    mask_right = tip_coord >= threshold
    mask_left = ~mask_right

    print(f"Threshold on axis {axis}: {threshold:.4f} | Right: {mask_right.sum().item()}, Left: {mask_left.sum().item()}")

    return {
        'right': (X[mask_right], Y[mask_right]),
        'left': (X[mask_left], Y[mask_left])
    }



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


def plot_generalization_results(
        results_csv: str,
        semipiani_npz: str,
        train_losses_kan: list,
        test_losses_kan: list = None,
        train_losses_mlp: list = None,
        test_losses_mlp: list = None,
        results_dir: str = "results"
):
    """
    Genera e salva nella cartella `results_dir`:
      1) Bar chart di RMSE e R² per semipiano (right/left)
      2) Scatter true vs pred per modello e semipiano
      3) Curva di convergenza loss vs epoca per train/test e modello (solo se disponibili)
    """
    os.makedirs(results_dir, exist_ok=True)

    # 1) Bar chart di RMSE e R²
    df = pd.read_csv(results_csv)
    for metric in ['RMSE', 'R2']:
        for side in df['side'].unique():
            sub = df[df['side'] == side].set_index('model')
            axes = ['X', 'Y', 'Z']
            vals = sub[[f'{ax}_{metric}' for ax in axes]].T
            vals.columns = ['KAN', 'MLP']
            plt.figure(figsize=(6, 4))
            vals.plot(kind='bar', width=0.7, ax=plt.gca())
            plt.title(f'{metric} per coordinata ({side})')
            plt.xlabel('Coordinata')
            plt.ylabel(metric)
            plt.tight_layout()
            png_path = os.path.join(results_dir, f'bar_{side}_{metric}.png')
            plt.savefig(png_path)
            plt.close()

    # 2) Scatter true vs pred
    data = np.load(semipiani_npz)
    sides = [key.split('_')[1] for key in data.files if key.startswith('true_')]
    models = ['kan', 'mlp']
    coord_labels = ['X', 'Y', 'Z']
    for side in sides:
        plt.figure(figsize=(12, 4))
        for mi, model in enumerate(models):
            for ci, coord in enumerate(coord_labels):
                key_true = f'true_{side}'
                key_pred = f'pred_{model}_{side}'
                y_true = data[key_true][:, ci]
                y_pred = data[key_pred][:, ci]
                ax = plt.subplot(2, 3, mi * 3 + ci + 1)
                ax.scatter(y_true, y_pred, alpha=0.3)
                mn, mx = y_true.min(), y_true.max()
                ax.plot([mn, mx], [mn, mx], '--', color='gray')
                ax.set_title(f'{model.upper()} {side} {coord}')
                ax.set_xlabel('True')
                ax.set_ylabel('Pred')
        plt.tight_layout()
        png_path = os.path.join(results_dir, f'scatter_{side}.png')
        plt.savefig(png_path)
        plt.close()

    # 3) Curva di convergenza
    if train_losses_kan and train_losses_mlp:
        epochs = list(range(1, len(train_losses_kan) + 1))
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_losses_kan, label='KAN train')
        if test_losses_kan:
            plt.plot(epochs, test_losses_kan, label='KAN test', linestyle='--')
        plt.plot(epochs, train_losses_mlp, label='MLP train')
        if test_losses_mlp:
            plt.plot(epochs, test_losses_mlp, label='MLP test', linestyle='--')
        plt.xlabel('Epoca')
        plt.ylabel('MSE Loss')
        plt.title('Convergenza')
        plt.legend()
        plt.grid(True)
        png_path = os.path.join(results_dir, 'convergenza_loss.png')
        plt.savefig(png_path)
        plt.close()


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
