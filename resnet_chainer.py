import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import numpy as np
import random

# 前処理
# 書き方はこれを参考にした：https://github.com/yasunorikudo/chainer-DenseNet
class Preprocess(chainer.dataset.DatasetMixin):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def get_example(self, i):
        x, y = self.pairs[i]
        # label
        y = np.array(y, dtype=np.int32)

        # random crop
        pad_x = np.zeros((3, 40, 40), dtype=np.float32)
        pad_x[:, 4:36, 4:36] = x
        top = random.randint(0, 8)
        left = random.randint(0, 8)
        x = pad_x[:, top:top+32, left:left+32]
        # horizontal flip
        if random.randint(0, 1):
            x = x[:, :, ::-1]

        return x, y

# Model
# ResBlock単体
class ResBlock(chainer.Chain):
    # BN->ReLU->Conv->BN->ReLU->Conv をショートカットさせる(Kaimingらの研究による)
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        w = chainer.initializers.GlorotNormal()
        
        with self.init_scope():
            self.bn1 = L.BatchNormalization(channels)
            self.conv1 = L.Convolution2D(None, channels, ksize=3, pad=1, initialW=w)
            self.bn2 = L.BatchNormalization(channels)
            self.conv2 = L.Convolution2D(None, channels, ksize=3, pad=1, initialW=w)

    def __call__(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + x

# オリジナルの論文に従って、サブサンプリングにPoolingではなくstride=2のConvを使う
class Subsumpling(chainer.Chain):
    def __init__(self, output_channels):
        super().__init__()
        w = chainer.initializers.GlorotNormal()
        
        with self.init_scope():
            self.conv = L.Convolution2D(None, output_channels, ksize=1, stride=2, initialW=w)

    def __call__(self, x):
        return self.conv(x)

class ResNet(chainer.Chain):
    def __init__(self, n):
        super().__init__()
        self.n = n
        w = chainer.initializers.GlorotNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 16, ksize=3, pad=1, initialW=w) #3->16
            self.rbs1 = self._make_resblocks(16, n)
            self.pool1 = Subsumpling(32)
            self.rbs2 = self._make_resblocks(32, n)
            self.pool2 = Subsumpling(64)
            self.rbs3 = self._make_resblocks(64, n)
            self.fc = L.Linear(None, 10, initialW=w)

    def _make_resblocks(self, channels, count):
        layers = [ResBlock(channels) for i in range(count)]
        return chainer.Sequential(*layers)

    def __call__(self, x):
        out = self.conv1(x)
        out = self.rbs1(out)
        out = self.pool1(out)
        out = self.rbs2(out)
        out = self.pool2(out)
        out = self.rbs3(out)
        out = F.average_pooling_2d(out, ksize=8) #最後は(8,8)
        out = self.fc(out)
        return out

def main(n, nb_epochs):
    train, test = chainer.datasets.get_cifar10()
    train = Preprocess(train)
    test = Preprocess(test)
    train_iter = chainer.iterators.SerialIterator(train, 128)
    test_iter = chainer.iterators.SerialIterator(test, 100, repeat=False, shuffle=False)

    ### Parameters
    device = 0 # -1:CPU, 0:GPU
    ###

    net = chainer.links.Classifier(ResNet(n))

    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(net)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

    updater = training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (nb_epochs, "epoch"), out=f"chainer_n{n}")

    val_interval = (1, "epoch")
    log_interval = (1, "epoch")

    # 学習率調整
    def lr_shift():
        if updater.epoch == int(nb_epochs*0.5) or updater.epoch == int(nb_epochs*0.75):
            optimizer.lr *= 0.1
        return optimizer.lr

    trainer.extend(extensions.Evaluator(test_iter, net, device=device), trigger=val_interval)
    trainer.extend(extensions.observe_value(
        "lr", lambda _: lr_shift()), trigger=(1, "epoch"))
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport([
        'elapsed_time', 'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'lr',
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=50))

    trainer.run()

if __name__ == "__main__":
    main(3, 1)