import os
import torch
import torch.nn as nn
import torch.optim as optim

from src.KAN_RNN.KAN_Rnn import KAN_Rnn
from src.utility import prepare_sequence_loaders

# ─── Config ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
lr = 1e-3
epochs = 50
seq_len = 20  # lunghezza finestra temporale
hidden_dim = 64

print(f"Using device: {device}")

# ─── 1) Dataloader sequenziale ────────────────────────────────────────────
train_loader, test_loader, in_dim, out_dim = prepare_sequence_loaders(
    deformation="bending",
    train_trial=1,
    test_trial=2,
    seq_len=seq_len,
    batch_size=batch_size
)

# ─── 2) Modello ───────────────────────────────────────────────────────────
layer_configs = [
    {"n_knots": 16, "x_min": -1.0, "x_max": 1.0, "use_bn": True, "activation": nn.ReLU(), "dropout": 0.1},
    {"n_knots": 12, "x_min": -1.0, "x_max": 1.0, "use_bn": False, "activation": nn.ReLU(), "dropout": 0.0},
]

model = KAN_Rnn(
    input_dim=in_dim,
    hidden_dim=hidden_dim,
    layer_configs=layer_configs,
    output_dim=out_dim
).to(device)

# ─── 3) Ottimizzatore e loss ───────────────────────────────────────────────
opt = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()
best_loss = float('inf')
best_path = "best_kan_rnn.pth"


# ─── 4) Funzioni di training / val (raw data) ───────────────────────────────
def train_one_epoch(m, loader, loss_fn, opt, device):
    m.train()
    total = 0.0
    for X_seq, Y_seq in loader:
        X_seq, Y_seq = X_seq.to(device), Y_seq.to(device)
        opt.zero_grad()
        Y_pred = m(X_seq)
        loss = loss_fn(Y_pred, Y_seq)
        loss.backward()
        opt.step()
        total += loss.item() * X_seq.size(0)
    return total / len(loader.dataset)


def eval_loss(m, loader, loss_fn, device):
    m.eval()
    total = 0.0
    with torch.no_grad():
        for X_seq, Y_seq in loader:
            X_seq, Y_seq = X_seq.to(device), Y_seq.to(device)
            total += loss_fn(m(X_seq), Y_seq).item() * X_seq.size(0)
    return total / len(loader.dataset)


# ─── 5) Loop di training ──────────────────────────────────────────────────
for ep in range(1, epochs + 1):
    tr_loss = train_one_epoch(model, train_loader, loss_fn, opt, device)
    ts_loss = eval_loss(model, test_loader, loss_fn, device)
    print(f"Ep {ep:02d} | train loss {tr_loss:.4f} | test loss {ts_loss:.4f}")
    if ts_loss < best_loss:
        best_loss = ts_loss
        torch.save(model.state_dict(), best_path)

# ─── 6) Risultati finali ───────────────────────────────────────────────────
print(f"\nBest test loss RNN ➔ {best_loss:.4f} (saved to {best_path})")
