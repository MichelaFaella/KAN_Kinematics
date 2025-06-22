import os
import json
import torch
import datetime
import torch.nn as nn
from src.KAN.KAN_Net import KAN_Net
from src.MLP.MLP_Net import MLP_Net
from src.data.loader import MyDataLoader
from src.utility import (
    split_dataset, create_dataloader,
    eval_loss, visualize_performance
)

# ----- Setup -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")
batch_size = 64

# ----- Caricamento Dati -----
dl = MyDataLoader()
dl.load_data(deformation="bending", trial_num=1)
data = dl.get_data()
X, Y = data["X"], data["Y"]

# ----- Split e Dataloader -----
X_train, X_test, Y_train, Y_test = split_dataset(X, Y, split_ratio=0.8)
train_loader = create_dataloader(X_train, Y_train, batch_size)
test_loader = create_dataloader(X_test, Y_test, batch_size)

# ----- KAN Config -----
layer_configs = [
    {"out_features": 64, "n_knots": 16, "x_min": -1.0, "x_max": 1.0, "use_bn": True, "dropout": 0.1},
    {"out_features": 32, "n_knots": 12, "x_min": -1.0, "x_max": 1.0, "use_bn": False, "dropout": 0.0},
]

# ----- Inizializza Modelli -----
kan = KAN_Net(input_dim=X.shape[1], layer_configs=layer_configs, output_dim=Y.shape[1]).to(device)
mlp = MLP_Net(input_dim=X.shape[1], hidden_dims=[64, 32], output_dim=Y.shape[1]).to(device)

# ----- Loss & Ottimizzatori -----
loss_fn = nn.MSELoss()
optimizer_kan = torch.optim.Adam(kan.parameters(), lr=1e-3, weight_decay=1e-5)
optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-5)

# ----- Training Loop -----
def train_model(model, loader, loss_fn, optimizer, name, device, epochs=50):
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{name}] Epoch {epoch:02d} - Loss: {total_loss:.6f}")

# ----- Addestramento -----
train_model(kan, train_loader, loss_fn, optimizer_kan, "KAN", device)
train_model(mlp, train_loader, loss_fn, optimizer_mlp, "MLP", device)

# ----- Valutazione -----
test_loss_kan = eval_loss(kan, test_loader, loss_fn, device)
test_loss_mlp = eval_loss(mlp, test_loader, loss_fn, device)
print(f"📊 KAN test loss: {test_loss_kan:.4f}")
print(f"📊 MLP test loss: {test_loss_mlp:.4f}")

# ----- Visualizzazione -----
visualize_performance(
    test_losses_kan=[test_loss_kan],
    test_losses_mlp=[test_loss_mlp],
    kan=kan,
    mlp=mlp,
    test_loader=test_loader,
    device=device
)

# ----- Salvataggio Risultati -----
plot_dir = os.path.join("plot_static", datetime.date.today().isoformat())
os.makedirs(plot_dir, exist_ok=True)
with open(os.path.join(plot_dir, "test_losses.json"), "w") as f:
    json.dump({
        "KAN_test_loss": test_loss_kan,
        "MLP_test_loss": test_loss_mlp
    }, f, indent=4)
print(f"✅ Test losses salvate in '{plot_dir}/test_losses.json'")
