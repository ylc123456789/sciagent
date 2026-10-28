'''Train CIFAR-10 ResNet18 on the fixed 45k train split (neutral baseline).

Neutral adaptation of kuangliu/pytorch-cifar for the L3 calibration case:
- ResNet18 (the case's fixed architecture)
- local dataset via RESAGENT2_DATASET_ROOT, download=False
- trains only on the first 45000 train samples (deterministic split; the last
  5000 are the calibration set, and the official test set is not loaded here)
- saves the checkpoint; no test evaluation
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import json
import argparse

from models import ResNet18
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 ResNet18 baseline (45k train split)')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', default=200, type=int, help='number of training epochs')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--output-dir', default='./checkpoint', type=str,
                    help='directory to save the checkpoint')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data (local, no download). Only the 45k train split is used here.
print('==> Preparing data..')
dataset_root = json.loads(os.environ.get('RESAGENT2_DATASETS_JSON', '{}')).get('cifar10', os.environ.get('RESAGENT2_DATASET_ROOT', './data'))
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

full_trainset = torchvision.datasets.CIFAR10(
    root=dataset_root, train=True, download=False, transform=transform_train)
# Deterministic split (see split.json): first 45000 = train, last 5000 = calibration.
trainset = torch.utils.data.Subset(full_trainset, list(range(45000)))
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


for epoch in range(args.epochs):
    train(epoch)
    scheduler.step()

# Save the final checkpoint (no test selection; model selection is the Agent's decision).
os.makedirs(args.output_dir, exist_ok=True)
ckpt_path = os.path.join(args.output_dir, 'ckpt.pth')
if isinstance(net, torch.nn.DataParallel):
    state_net = net.module.state_dict()
else:
    state_net = net.state_dict()
torch.save({'net': state_net, 'epoch': args.epochs - 1, 'arch': 'resnet18'}, ckpt_path)
print('Saved checkpoint to', ckpt_path)
