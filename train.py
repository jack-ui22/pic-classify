import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt
import argparse
from  torchvision import datasets, transforms
from model import Net

# parse
parser = argparse.ArgumentParser(description='PyTorch classify example')
parser.add_argument('--train_dir', type=str, default='./cat_dog/train')
parser.add_argument('--test_dir', type=str, default='./cat_dog/test')
parser.add_argument('--model_dir', type=str, default='best_model.pth')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
# parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args()
# 参数配置
#数据集
train_trans=transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomCrop(224),#随机裁剪
    transforms.ToTensor(),
])
test_trans=transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
train_dataset = datasets.ImageFolder(args.train_dir, transform=train_trans)
test_dataset = datasets.ImageFolder(args.test_dir, transform=test_trans)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          num_workers=2)
# device
model = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("using device:", device)
print("model:",model)
#损失函数和优化器
loss_fc=nn.CrossEntropyLoss()

test_losses=[]
train_losses=[]
test_accs=[]
train_accs=[]
best_test_acc = 0.0

for epoch in range(1, args.epochs + 1):
    if epoch <15:
        lr=0.001
    elif epoch < 45:
        lr=0.0005
    elif epoch < 75:
        lr=0.0003
    else:
        lr=0.0001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    train_total_loss = 0.0
    train_total_correct = 0
    train_total_samples = 0
    for x, y in tqdm(train_loader, desc=f"Train: {epoch}/{args.epochs}"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        y_grap = model(x)
        loss = loss_fc(y_grap, y)
        loss.backward()
        optimizer.step()
        batch_size = x.size(0)
        train_total_loss += loss.item() * batch_size
        y_pred = y_grap.argmax(dim=1)
        train_total_correct += (y_pred == y).sum().item()
        train_total_samples += batch_size

    train_avg_loss = train_total_loss / train_total_samples
    train_avg_acc = train_total_correct / train_total_samples
    train_losses.append(train_avg_loss)
    train_accs.append(train_avg_acc)
    model.eval()
    test_total_loss = 0.0
    test_total_correct = 0
    test_total_samples = 0

    with torch.no_grad():  #
        for x, y in tqdm(test_loader, desc=f"Test:  {epoch}/{args.epochs}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_grap = model(x)
            loss = loss_fc(y_grap, y)
            batch_size = x.size(0)
            test_total_loss += loss.item() * batch_size
            y_pred = y_grap.argmax(dim=1)
            test_total_correct += (y_pred == y).sum().item()
            test_total_samples += batch_size

    test_avg_loss = test_total_loss / test_total_samples
    test_avg_acc = test_total_correct / test_total_samples
    test_losses.append(test_avg_loss)
    test_accs.append(test_avg_acc)

    if epoch % 10 == 0:
        print(f"\nEpoch [{epoch}/{args.epochs}]")
        print(f"Train - Loss: {train_avg_loss:.4f}, Acc: {train_avg_acc:.4f}")
        print(f"Test  - Loss: {test_avg_loss:.4f}, Acc: {test_avg_acc:.4f}")

    if test_avg_acc > best_test_acc:
        best_test_acc = test_avg_acc
        torch.save(model.state_dict(), args.model_dir)
        print(f"\nEpoch {epoch} Test Acc improved to {best_test_acc:.4f}, save best model to {args.model_dir}\n")

plt.plot(train_losses,label='train')
plt.plot(test_losses,label='test')
plt.show()
plt.plot(train_accs,label='train')
plt.plot(test_accs,label='test')
plt.show()