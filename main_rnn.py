# main_train_rnn.py

import os
import torch
import torch.nn as nn
import torch.optim as optim

from src.KAN_RNN.KAN_Rnn import KAN_Rnn
from src.utility import train_one_epoch, eval_loss, prepare_sequence_loaders

# ─── Setup ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 32
lr = 1e-3
epochs = 50
seq_len = 10  # lunghezza finestra temporale
hidden_dim = 64

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
  {"n_knots": 16, "x_min": -1., "x_max": 1.,
   "use_bn": True, "activation": nn.ReLU(), "dropout": 0.1},
  {"n_knots": 12, "x_min": -1., "x_max": 1.,
   "use_bn": False, "activation": nn.ReLU(), "dropout": 0.0},
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

# ─── 4) Loop di training ──────────────────────────────────────────────────
for ep in range(1, epochs + 1):
    # train
    tr_loss = train_one_epoch(model, train_loader, loss_fn, opt, device)
    # val
    ts_loss = eval_loss(model, test_loader, loss_fn, device)

    print(f"Ep {ep:02d} | train loss {tr_loss:.4f} | test loss {ts_loss:.4f}")

    # salva il migliore su test
    if ts_loss < best_loss:
        best_loss = ts_loss
        torch.save(model.state_dict(), best_path)

# ─── 5) Risultati finali ───────────────────────────────────────────────────
print(f"\nBest test loss RNN ➔ {best_loss:.4f} (saved to {best_path})")
