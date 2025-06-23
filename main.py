import datetime
import json
import os
import torch
import torch.nn as nn

from src.KAN.KAN_Net import KAN_Net
from src.MLP.MLP_Net import MLP_Net
from src.utility import (
    prepare_loaders, prepare_sequence_loaders,
    eval_loss, visualize_performance, plot_kan_splines
)

# ----- Configurazione -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")
batch_size = 64
seq_len = 10
hidden_dim = 64

# ----- Dataloaders -----
# Restituiscono: (input = attuazioni [N, 9], output = marker finale [N, 3])
train_loader, test_loader = prepare_loaders("bending", 1, 2, batch_size)
_, seq_test_loader, in_dim, out_dim = prepare_sequence_loaders(
    deformation="bending",
    train_trial=1,
    test_trial=2,
    seq_len=seq_len,
    batch_size=batch_size
)

# Controllo shape degli input/output
print(f"📥 Input dim:  {in_dim} (aspettato: 9)")
print(f"📤 Output dim: {out_dim} (aspettato: 3)")

# ----- Configurazione KAN -----
layer_configs = [
    {
        "out_features": 4,
        "n_knots": 6,
        "x_min": -1.0,
        "x_max": 1.0,
        "use_bn": False,
        "dropout": 0.0
    },
]

# ----- Inizializza Modelli -----
kan = KAN_Net(input_dim=in_dim, layer_configs=layer_configs, output_dim=out_dim).to(device)
mlp = MLP_Net(input_dim=in_dim, hidden_dims=[64, 32], output_dim=out_dim).to(device)

# ----- Loss e Ottimizzatori -----
loss_fn = nn.MSELoss()
optimizer_kan = torch.optim.Adam(kan.parameters(), lr=1e-3, weight_decay=1e-5)
optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-5)

# ----- Funzione di Training -----
def train_model(model, train_loader, loss_fn, optimizer, device, epochs=50):
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{model.__class__.__name__}] Epoch {epoch:02d} - Loss: {total_loss:.6f}")

# ----- Addestramento -----
train_model(kan, train_loader, loss_fn, optimizer_kan, device)
train_model(mlp, train_loader, loss_fn, optimizer_mlp, device)

# ----- Valutazione -----
l_ts_kan = eval_loss(kan, test_loader, loss_fn, device)
l_ts_mlp = eval_loss(mlp, test_loader, loss_fn, device)
print(f"📊 Static KAN test loss: {l_ts_kan:.4f}")
print(f"📊 Static MLP test loss: {l_ts_mlp:.4f}")

# ----- Visualizzazione -----
visualize_performance(
    test_losses_kan=[l_ts_kan],
    test_losses_mlp=[l_ts_mlp],
    kan=kan,
    mlp=mlp,
    test_loader=test_loader,
    device=device
)

# ----- Salvataggio Risultati -----
plot_dir = os.path.join("plots/plot_rnn", datetime.date.today().isoformat())
os.makedirs(plot_dir, exist_ok=True)
with open(os.path.join(plot_dir, "test_losses.json"), "w") as f:
    json.dump({"KAN_test_loss": l_ts_kan, "MLP_test_loss": l_ts_mlp}, f, indent=4)
print(f"✅ Saved test losses to {os.path.join(plot_dir, 'test_losses.json')}")

# ----- Visualizza spline KAN -----
plot_kan_splines(kan)
