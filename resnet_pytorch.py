import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import time
import pickle

# DataLoader
class Loader:
    def __init__(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        self.train = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        self.test = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

# Model
# ResBlock単体
class ResBlock(nn.Module):
    # BN->ReLU->Conv->BN->ReLU->Conv をショートカットさせる(Kaimingらの研究による)
    def __init__(self, channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        return out + x

# オリジナルの論文に従って、サブサンプリングにPoolingではなくstride=2のConvを使う
class Subsumpling(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=2)

    def forward(self, x):
        return self.conv(x)

class ResNet(nn.Module):
    def __init__(self, n, initial_lr=0.01, nb_epochs=100):
        super().__init__()
        self.n = n
        self.initial_lr = initial_lr
        self.nb_epochs = nb_epochs
        self.weight_decay = 0.0005
        # Model
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1) #3->16
        self.rbs1 = self._make_resblocks(16, n)
        self.pool1 = Subsumpling(16, 32)
        self.rbs2 = self._make_resblocks(32, n)
        self.pool2 = Subsumpling(32, 64)
        self.rbs3 = self._make_resblocks(64, n)
        self.gap = nn.AvgPool2d(kernel_size=8) # (8,8)をGlobal average pooling
        self.fc = nn.Linear(64, 10)
        # 履歴
        self.history = {"loss":[], "acc":[], "val_loss":[], "val_acc":[], "time":[]}

    def _make_resblocks(self, channels, count):
        layers = [ResBlock(channels) for i in range(count)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.rbs1(out)
        out = self.pool1(out)
        out = self.rbs2(out)
        out = self.pool2(out)
        out = self.rbs3(out)
        out = self.gap(out)
        out = out.view(-1, 64)
        out = self.fc(out)
        return out

    def compile(self, device):
        assert device in ["cuda", "cpu"]
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, momentum=0.9)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[self.nb_epochs*0.5, self.nb_epochs*0.75], gamma=0.1)

    #Train
    def fit_train(self, loader, epoch):
        print('\nEpoch: %d' % epoch)
        self.train()
        train_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(loader.train):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx%50 == 0:
                print(batch_idx, len(loader.train), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        self.history["loss"].append(train_loss/(batch_idx+1))
        self.history["acc"].append(1.*correct/total)
        self.history["time"].append(time.time()-start_time)

    def fit_validate(self, loader, epoch):
        self.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader.test):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self(inputs)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc = 100.*correct/total

        print(batch_idx, len(loader.test), 'ValLoss: %.3f | ValAcc: %.3f%% (%d/%d)'
            % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))
        self.history["val_loss"].append(val_loss/(batch_idx+1))
        self.history["val_acc"].append(1.*correct/total)

    def fit(self, loader):
        for epoch in range(self.nb_epochs):
            self.scheduler.step()
            self.fit_train(loader, epoch)
            self.fit_validate(loader, epoch)

    def save_history(self):
        file_name = f"pytorch_n{self.n}.dat"
        with open(file_name, "wb") as fp:
            pickle.dump(self.history, fp)

def main(n, nb_epochs=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark=True
    # Data
    data = Loader()
    # Create Model
    net = ResNet(n, nb_epochs=nb_epochs)
    if device=="cuda":
        net = net.cuda()
    # Compile
    net.compile(device)
    # fit
    net.fit(data)
    # savehistory
    net.save_history()

if __name__ == "__main__":
    main(3, 1)