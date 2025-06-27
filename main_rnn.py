# Training script for KAN, MLP, and KAN_Rnn
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from src.KAN.KAN_Net import KAN_Net
from src.KAN_RNN.KAN_Rnn import KAN_Rnn
from src.MLP.MLP_Net import MLP_Net
from src.utility import train_one_epoch, eval_loss, visualize_performance_rnn, plot_rnn_predictions_workspace
from dataset.data_loader import DataLoader as MyDataLoader

# ─── Config ─────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
lr = 1e-3
epochs = 50
seq_len = 20
hidden_dim = 64
print(f"Using device: {device}")

# ─── Load Data ──────────────────────────────────
dl_train = MyDataLoader()
dl_train.load_data(deformation="bending", trial_num=1)
data_train = dl_train.get_data()
X_train = torch.tensor(data_train['actuation'], dtype=torch.float32)
Y_train = torch.tensor(data_train['markers'][:, -1, :], dtype=torch.float32)


dl_test = MyDataLoader()
dl_test.load_data(deformation="bending", trial_num=2)
data_test = dl_test.get_data()
X_test = torch.tensor(data_test['actuation'], dtype=torch.float32)
Y_test = torch.tensor(data_test['markers'][:, -1, :], dtype=torch.float32)

in_dim = X_train.shape[1]
out_dim = Y_train.shape[1]

# Static loaders
static_train_loader = TorchDataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
static_test_loader = TorchDataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False)

# Sequential loaders
def create_sequence_loader(X, Y, seq_len, batch_size, shuffle):
    sequences, targets = [], []
    for i in range(len(X) - seq_len):
        sequences.append(X[i:i+seq_len])
        targets.append(Y[i+seq_len - 1])
    return TorchDataLoader(
        TensorDataset(torch.stack(sequences), torch.stack(targets)),
        batch_size=batch_size,
        shuffle=shuffle
    )

seq_train_loader = create_sequence_loader(X_train, Y_train, seq_len, batch_size, shuffle=True)
seq_test_loader = create_sequence_loader(X_test, Y_test, seq_len, batch_size, shuffle=False)

# ─── Build Models ───────────────────────────────
layer_configs = [
    {"out_features": 64, "n_knots": 16, "x_min": -1.0, "x_max": 1.0, "use_bn": True, "dropout": 0.1},
    {"out_features": 32, "n_knots": 12, "x_min": -1.0, "x_max": 1.0, "use_bn": False, "dropout": 0.0},
]

kan = KAN_Net(input_dim=in_dim, layer_configs=layer_configs, output_dim=out_dim).to(device)
mlp = MLP_Net(input_dim=in_dim, hidden_dims=[64, 32], output_dim=out_dim).to(device)
rnn = KAN_Rnn(input_dim=in_dim, hidden_dim=hidden_dim, layer_configs=layer_configs, output_dim=out_dim).to(device)

models = {"KAN": kan, "MLP": mlp, "KAN_RNN": rnn}
optimizers = {name: optim.Adam(model.parameters(), lr=lr) for name, model in models.items()}
loss_fn = nn.MSELoss()
best_losses = {name: float('inf') for name in models}
best_paths = {name: f"models/best_{name.lower()}.pth" for name in models}
os.makedirs("models", exist_ok=True)

# ─── Training Loop ──────────────────────────────
for ep in range(1, epochs + 1):
    print(f"\nEpoch {ep}/{epochs}")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        if name == "KAN_RNN":
            train_loss = train_one_epoch(model, seq_train_loader, loss_fn, optimizers[name], device)
            test_loss = eval_loss(model, seq_test_loader, loss_fn, device)
        else:
            train_loss = train_one_epoch(model, static_train_loader, loss_fn, optimizers[name], device)
            test_loss = eval_loss(model, static_test_loader, loss_fn, device)

        print(f"{name} ➜ Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f}")

        if test_loss < best_losses[name]:
            best_losses[name] = test_loss
            torch.save(model.state_dict(), best_paths[name])
            print(f"✅ {name}: New best model saved (loss: {test_loss:.4f})")

# ─── Load Best Weights ──────────────────────────
for name, model in models.items():
    model.load_state_dict(torch.load(best_paths[name], map_location=device))

# ─── Final Evaluation & Visualization ───────────
final_kan_loss = eval_loss(kan, static_test_loader, loss_fn, device)
final_mlp_loss = eval_loss(mlp, static_test_loader, loss_fn, device)
final_rnn_loss = eval_loss(rnn, seq_test_loader, loss_fn, device)

visualize_performance_rnn(
    test_losses_kan=[final_kan_loss],
    test_losses_mlp=[final_mlp_loss],
    test_losses_rnn=[final_rnn_loss],
    kan_model=kan,
    mlp_model=mlp,
    rnn_model=rnn,
    static_loader=static_test_loader,
    seq_loader=seq_test_loader,
    device=device
)
