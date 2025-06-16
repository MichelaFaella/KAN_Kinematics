# Eval models only, no training
import datetime
import json
import os
import torch
import torch.nn as nn
import numpy as np
from src.KAN.KAN_Net import KAN_Net
from src.KAN_RNN.KAN_Rnn import KAN_Rnn
from src.MLP.MLP_Net import MLP_Net
from src.utility import prepare_loaders, eval_loss, visualize_performance, prepare_sequence_loaders, \
    visualize_performance_rnn

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
batch_size = 64
seq_len = 10
hidden_dim = 64

# 1) Prepara dataloader
train_loader, test_loader = prepare_loaders("bending", 1, 2, batch_size)
_, seq_test_loader, in_dim, out_dim = prepare_sequence_loaders(
    deformation="bending",
    train_trial=1,
    test_trial=2,
    seq_len=seq_len,
    batch_size=batch_size
)

# 2) Configurazione KAN
layer_configs = [
    {"out_features": 64, "n_knots": 16, "x_min": -1.0, "x_max": 1.0,
     "use_bn": True, "activation": nn.ReLU(), "dropout": 0.1},
    {"out_features": 32, "n_knots": 12, "x_min": -1.0, "x_max": 1.0,
     "use_bn": False, "activation": nn.ReLU(), "dropout": 0.0},
]

# 3) Instanzia modelli e carica pesi
kan = KAN_Net(input_dim=9, layer_configs=layer_configs, output_dim=3).to(device)
mlp = MLP_Net(input_dim=9, hidden_dims=[64, 32], output_dim=3).to(device)
rnn = KAN_Rnn(
    input_dim=in_dim,
    hidden_dim=hidden_dim,
    layer_configs=[
        {"n_knots": 16, "x_min": -1.0, "x_max": 1.0,
         "use_bn": True, "activation": nn.ReLU(), "dropout": 0.1},
        {"n_knots": 12, "x_min": -1.0, "x_max": 1.0,
         "use_bn": False, "activation": nn.ReLU(), "dropout": 0.0},
    ],
    output_dim=out_dim
).to(device)

kan.load_state_dict(torch.load("best_kan.pth", map_location=device))
mlp.load_state_dict(torch.load("best_mlp.pth", map_location=device))
rnn.load_state_dict(torch.load("best_kan_rnn.pth", map_location=device))

# 4) Loss function
loss_fn = torch.nn.MSELoss()

# 5) Valutazione solo test
l_ts_kan = eval_loss(kan, test_loader, loss_fn, device)
l_ts_mlp = eval_loss(mlp, test_loader, loss_fn, device)
l_ts_rnn = eval_loss(rnn, seq_test_loader, loss_fn, device)

print(f"Static KAN test loss: {l_ts_kan:.4f}")
print(f"Static MLP test loss: {l_ts_mlp:.4f}")
print(f"Dynamic RNN test loss: {l_ts_rnn:.4f}\n")

# 6) Visualizza e salva nella cartella plot
"""visualize_performance(
    test_losses_kan=[l_ts_kan],
    test_losses_mlp=[l_ts_mlp],
    kan=kan,
    mlp=mlp,
    test_loader=test_loader,
    device=device
)"""

visualize_performance_rnn(
    test_losses_kan=[l_ts_kan],
    test_losses_mlp=[l_ts_mlp],
    test_losses_rnn=[l_ts_rnn],
    kan_model=kan,
    mlp_model=mlp,
    rnn_model=rnn,
    static_loader=test_loader,
    seq_loader=seq_test_loader,
    device=device
)

# 7) Salva le perdite in JSON
plot_dir = os.path.join("plot_rnn", datetime.date.today().isoformat())
os.makedirs(plot_dir, exist_ok=True)
with open(os.path.join(plot_dir, "test_losses.json"), "w") as f:
    json.dump({"KAN_test_loss": l_ts_kan, "MLP_test_loss": l_ts_mlp, "RNN_KAN_test_loss": l_ts_rnn}, f, indent=4)
print(f"Saved test losses to {os.path.join(plot_dir, 'test_losses.json')}")
