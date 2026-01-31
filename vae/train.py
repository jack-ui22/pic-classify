import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from model import vae
from data import train_dataset, test_dataset
import matplotlib.pyplot as plt
from torch.utils.data import  DataLoader

# arg
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--mpath', type=str, default="best_model.pth")
parser.add_argument('--model', type=str, default="vae", choices=["vae", "gan"])
arg = parser.parse_args()
#loss_fn
model=vae()
model.init_weights()
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=arg.batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=False
)
test_dataloader = DataLoader(
    dataset=test_dataset,

    batch_size=arg.batch_size,
    shuffle=False,
    num_workers=0
)

optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr)
#送到gpu
device=torch.device("cuda"if torch.cuda.is_available() else "cpu")
model.to(device)
#loss
def vae_loss(z, x, mu, logvar):
    x = x.view(x.size(0), -1)
    z = z.view(z.size(0), -1)
    BCE = F.binary_cross_entropy(z, x, reduction='mean')
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
#数据记录

train_losses = []
test_losses = []
#训练
best_loss = float("inf")

for epoch in range(arg.epochs):
    train_loss=0.0
    loss = 0.0
    total = 0
    model.train()
    train_bar = tqdm(train_dataloader, desc=f"Train Epoch {epoch:03d}/{arg.epochs:03d}")
    for train_x,labals in train_bar:
        optimizer.zero_grad()
        train_x = train_x.to(device).float()
        z, mu, logvar = model(train_x,labals)
        loss = vae_loss(z, train_x, mu, logvar)
        train_loss += loss.item()
        total += train_x.size(0)
        loss.backward()
        optimizer.step()
    train_losses.append(train_loss/len(train_dataloader))
    test_loss=0.0
    total=0
    loss=0.0
    model.eval()
    test_bar=tqdm(test_dataloader,desc=f"Test Epoch {epoch:03d}/{arg.epochs:03d}")
    with torch.no_grad():
        for test_x,labals in test_bar:
            test_x = test_x.to(device).float()
            z, mu, logvar = model(test_x,labals)
            loss = vae_loss(z, test_x, mu, logvar)
            test_loss += loss.item()
            total += test_x.size(0)

    test_losses.append(test_loss/len(test_dataloader))
    print(f"\nTrain Loss:{train_loss/len(train_dataloader):.04f}",f"Test Loss:{test_loss/len(test_dataloader):.04f}")
    if test_loss < best_loss:
        print(f"Saving best model loss :{test_loss/len(test_dataloader):.04f}")
        torch.save(model.state_dict(), arg.mpath)
        best_loss = test_loss
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.show()


