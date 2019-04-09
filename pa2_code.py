'''

Data Split
Use train_dataset and eval_dataset as train / test sets

'''
from torchvision.datasets import EMNIST
from torch.utils.data import ConcatDataset, Subset, DataLoader
from torchvision.transforms import ToTensor, Compose
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import numpy as np
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from torch.optim import Adam, SGD  # just choose which to use
import os


# For convenience, show image at index in dataset
def show_image(dataset, index):
    import matplotlib.pyplot as plt
    plt.imshow(dataset[index][0][0], cmap=plt.get_cmap('gray'))


def get_datasets(split='balanced', save=False):
    download_folder = './data'

    transform = Compose([ToTensor()])

    dataset = ConcatDataset([EMNIST(root=download_folder, split=split, download=True, train=False, transform=transform),
                             EMNIST(root=download_folder, split=split, download=True, train=True, transform=transform)])

    # Ignore the code below with argument 'save'
    if save:
        random_seed = 4211  # do not change
        n_samples = len(dataset)
        eval_size = 0.2
        indices = list(range(n_samples))
        split = int(np.floor(eval_size * n_samples))

        np.random.seed(random_seed)
        np.random.shuffle(indices)

        train_indices, eval_indices = indices[split:], indices[:split]

        # cut to half
        train_indices = train_indices[:len(train_indices) // 2]
        eval_indices = eval_indices[:len(eval_indices) // 2]

        np.savez('train_test_split.npz', train=train_indices, test=eval_indices)

    # just use save=False for students
    # load train test split indices
    else:
        with np.load('./train_test_split.npz') as f:
            train_indices = f['train']
            eval_indices = f['test']

    train_dataset = Subset(dataset, indices=train_indices)
    eval_dataset = Subset(dataset, indices=eval_indices)

    return train_dataset, eval_dataset


# TODO
# 1. build your own CNN classifier with the given structure. DO NOT COPY OR USE ANY TRICK

def train_and_validate(model, loaders, optimizer, writer, n_epochs, ckpt_path, device='cpu'):
    def run_epoch(train_or_eval):
        epoch_loss = 0.
        epoch_acc = 0.
        epoch_acc_top_3 = 0.
        for i, batch in enumerate(loaders[train_or_eval], 1):
            in_data, labels = batch
            in_data, labels = in_data.to(device), labels.to(device)

            if train_or_eval == 'train':
                optimizer.zero_grad()

            logits = model(in_data)
            batch_loss = model.loss(logits, labels)
            batch_acc = model.top1_accuracy(logits, labels)
            batch_acc_top_3 = model.top3_accuracy(logits, labels)

            epoch_loss += batch_loss.item()
            epoch_acc += batch_acc.item()
            epoch_acc_top_3 += batch_acc_top_3.item()

            if train_or_eval == 'train':
                batch_loss.backward()
                optimizer.step()

        epoch_loss /= i
        epoch_acc /= i
        epoch_acc_top_3 /= i

        losses[train_or_eval] = epoch_loss
        accs[train_or_eval] = epoch_acc
        accs_top_3[train_or_eval] = epoch_acc_top_3

        if writer is None:
            print('epoch %d %s loss %.4f acc %.4f' % (epoch, train_or_eval, epoch_loss, epoch_acc))
        elif train_or_eval == 'eval':
            writer.add_scalars('%s_loss' % model.__class__.__name__,  # CnnClassifier or FcClassifier
                               tag_scalar_dict={'train': losses['train'],
                                                'eval': losses['eval']},
                               global_step=epoch)

            writer.add_scalars('%s_top1_accuracy' % model.__class__.__name__,  # CnnClassifier or FcClassifier
                               tag_scalar_dict={'train': accs['train'],
                                                'eval': accs['eval']},
                               global_step=epoch)
            writer.add_scalars('%s_top3_accuracy' % model.__class__.__name__,  # CnnClassifier or FcClassifier
                               tag_scalar_dict={'train': accs_top_3['train'],
                                                'eval': accs_top_3['eval']},
                               global_step=epoch)

            # For instructional purpose, add images here, just the last in_data
            if epoch % 10 == 0:
                if len(in_data.size()) == 2:  # when it is flattened, reshape it
                    in_data = in_data.view(-1, 1, 28, 28)

                img_grid = make_grid(in_data.to('cpu'))
                writer.add_image('%s/eval_input' % model.__class__.__name__, img_grid, epoch)

    # main statements
    losses = dict()
    accs = dict()
    accs_top_3 = dict()

    for epoch in range(1, n_epochs + 1):
        run_epoch('train')
        run_epoch('eval')

        # For instructional purpose, show how to save checkpoints
        if ckpt_path is not None:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'losses': losses,
                'accs': accs,
                'accs_top_3': accs_top_3
            }, '%s/%d.pt' % (ckpt_path, epoch))


# Modified from Tutorial 6
# Define CNN classifier

class MyCnnClassifier(nn.Module):
    # n_hidden: number of units at the last fc layer
    def __init__(self, n_hidden):
        super(MyCnnClassifier, self).__init__()

        # in_data size: (batch_size, 1, 28, 28)
        self.cnn_layers = nn.Sequential(
            # conv1_out size: (batch_size, 4, 26, 26)
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # conv2_out size: (batch_size, 8, 12, 12)
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            # conv3_out size: (batch_size, 16, 5, 5)
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            # maxPool_out size: (batch_size, 16, 3, 3)
            nn.MaxPool2d(3, stride=1, padding=0),
            # conv4_out size: (batch_size, 32, 1, 1)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid()
        )

        # linear layers transforms flattened image features into logits before the softmax layer
        self.linear = nn.Sequential(
            nn.Linear(32, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 47)  # there are 47 classes
        )

        self.softmax = nn.Softmax(dim=1)
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')  # will be divided by batch size

    def forward(self, in_data):
        img_features = self.cnn_layers(in_data).view(in_data.size(0), 32)  # in_data.size(0) == batch_size
        logits = self.linear(img_features)
        return logits

    def loss(self, logits, labels):
        preds = self.softmax(logits)  # size (batch_size, 47)
        return self.loss_fn(preds, labels) / logits.size(0)  # divided by batch_size

    """ Obtained from https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b"""
    def accuracy(self, output, target, topk=(3,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    # Top - 1 accuracy
    def top1_accuracy(self, logits, labels):
        return self.accuracy(self, logits, labels)[0]  # in percentage

    def top3_accuracy(self, logits, labels):
        return self.accuracy(self, logits, labels)[2]

# 2. load pretrained encoder from 'pretrained_encoder.pt' and build a CNN classifier on top of the encoder
class PretrainedCNNClassifier(nn.Module):
    def __init__(self, pretrained_model, n_hidden):
        super(PretrainedCNNClassifier, self).__init__()
        self.pretrained_model = pretrained_model
        self.linear = nn.Sequential(
            nn.Linear(32, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 47)  # there are 47 classes
        )
        self.softmax = nn.Softmax(dim=1)
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')  # will be divided by batch size

    def forward(self, in_data):
        img_features = self.pretrained_model.forward(in_data)  # in_data.size(0) == batch_size
        logits = self.linear(img_features)
        return logits

    def loss(self, logits, labels):
        preds = self.softmax(logits)  # size (batch_size, 47)
        return self.loss_fn(preds, labels) / logits.size(0)  # divided by batch_size

    """ Obtained from https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b"""

    def accuracy(self, output, target, topk=(3,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    # Top - 1 accuracy
    def top1_accuracy(self, logits, labels):
        return self.accuracy(self, logits, labels)[0]  # in percentage

    def top3_accuracy(self, logits, labels):
        return self.accuracy(self, logits, labels)[2]

# 3. load pretrained encoder from 'pretrained_encoder.pt' and build a Convolutional Autoencoder on top of the encoder (just need to implement decoder)
# *** Note that all the above tasks include implementation, training, analyzing, and reporting

# example main code
# each img has size (1, 28, 28) and each label is in {0, ..., 46}, a total of 47 classes
if __name__ == '__main__':
    train_ds, eval_ds = get_datasets()

    img_index = 10
    show_image(train_ds, img_index)
    show_image(eval_ds, img_index)

    parser = ArgumentParser()
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--ckpt', type=str, default='./ckpt/cnn')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--optim', type=str, default='adam')
    args = parser.parse_args()
    n_hidden = args.hidden
    gpu = args.gpu
    lr = args.lr
    batch_size = args.batch
    ckpt_path = args.ckpt
    n_epochs = args.epoch
    opt_str = args.optim

    ckpt_path = '%s/%s' % (ckpt_path, opt_str)

    if ckpt_path is not None:
        if not (os.path.exists(ckpt_path)):
            os.makedirs(ckpt_path)

    DEVICE = 'cpu'
    if gpu == -1:
        DEVICE = 'cpu'
    elif torch.cuda.is_available():
        DEVICE = gpu

    scratchModel = MyCnnClassifier(n_hidden).to(DEVICE)
    dataloaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, drop_last=False, shuffle=True),
        'eval': DataLoader(eval_ds, batch_size=batch_size, drop_last=False)
    }

    opt_class = None
    if opt_str == 'adam':
        opt_class = Adam
    elif opt_str == 'sgd':
        opt_class = SGD

    optimizer = opt_class(scratchModel.parameters(), lr=lr)
    writer = SummaryWriter('./logs/cnn/%s' % opt_str)

    train_and_validate(scratchModel, dataloaders, optimizer, writer, n_epochs, ckpt_path, DEVICE)
