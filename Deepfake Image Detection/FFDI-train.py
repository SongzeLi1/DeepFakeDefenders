import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn as nn
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch import optim
from tqdm import tqdm
import os
import math
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torchvision
import argparse
from os.path import join
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class MyDataset(Dataset):
    def __init__(self, txt, mode, path):
        self.img_path = []
        self.label = []
        self.path = path

        with open(txt, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip().split(',')[0] == 'img_name':
                    continue

                self.img_path.append(line.strip().split(',')[0])
                self.label.append(line.strip().split(',')[1])

        if mode == "train":
            self.transform = transforms.Compose([
                transforms.Resize((299, 299)),
                # transforms.RandomCrop((256, 256)),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize([0.5] * 3, [0.5] * 3)
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                # transforms.Normalize([0.5] * 3, [0.5] * 3)
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        # 测试
        self.img_path = self.img_path  #[8000:18000]
        self.label = self.label  #[8000:18000]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        image_path = os.path.join(self.path, self.img_path[i])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(int(self.label[i]), dtype=torch.long)

        return image, label


pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975
            # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = self.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def xception(num_classes=1000, pretrained='imagenet'):
    model = Xception(num_classes=num_classes)
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model = Xception(num_classes=num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    # TODO: ugly
    model.last_linear = model.fc
    del model.fc
    return model


def return_pytorch04_xception(pretrained=True):
    # Raises warning "src not broadcastable to dst" but thats fine
    model = xception(pretrained=False)
    if pretrained:
        # Load model in torch 0.4+
        model.fc = model.last_linear
        del model.last_linear
        state_dict = torch.load(
            "./checkpoint/xception-b5690688.pth")
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        model.load_state_dict(state_dict)
        model.last_linear = model.fc
        del model.fc
    return model


class TransferModel(nn.Module):
    """
    Simple transfer learning model that takes an imagenet pretrained model with
    a fc layer as base model and retrains a new fc layer for num_out_classes
    """

    def __init__(self, modelchoice, num_out_classes=2, dropout=0.0):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice
        if modelchoice == 'xception':
            self.model = return_pytorch04_xception()
            # Replace fc
            num_ftrs = self.model.last_linear.in_features
            if not dropout:
                self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
            else:
                print('Using dropout', dropout)
                self.model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        elif modelchoice == 'resnet50' or modelchoice == 'resnet18':
            if modelchoice == 'resnet50':
                self.model = torchvision.models.resnet50(pretrained=True)
            if modelchoice == 'resnet18':
                self.model = torchvision.models.resnet18(pretrained=True)
            # Replace fc
            num_ftrs = self.model.fc.in_features
            if not dropout:
                self.model.fc = nn.Linear(num_ftrs, num_out_classes)
            else:
                self.model.fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        else:
            raise Exception('Choose valid model, e.g. resnet50')

    def set_trainable_up_to(self, boolean, layername="Conv2d_4a_3x3"):
        """
        Freezes all layers below a specific layer and sets the following layers
        to true if boolean else only the fully connected final layer
        :param boolean:
        :param layername: depends on network, for inception e.g. Conv2d_4a_3x3
        :return:
        """
        # Stage-1: freeze all the layers
        if layername is None:
            for i, param in self.model.named_parameters():
                param.requires_grad = True
                return
        else:
            for i, param in self.model.named_parameters():
                param.requires_grad = False
        if boolean:
            # Make all layers following the layername layer trainable
            ct = []
            found = False
            for name, child in self.model.named_children():
                if layername in ct:
                    found = True
                    for params in child.parameters():
                        params.requires_grad = True
                ct.append(name)
            if not found:
                raise Exception('Layer not found, cant finetune!'.format(
                    layername))
        else:
            if self.modelchoice == 'xception':
                # Make fc trainable
                for param in self.model.last_linear.parameters():
                    param.requires_grad = True

            else:
                # Make fc trainable
                for param in self.model.fc.parameters():
                    param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x


def model_selection(modelname, num_out_classes,
                    dropout=None):
    """
    :param modelname:
    :return: model, image size, pretraining<yes/no>, input_list
    """
    if modelname == 'xception':
        return TransferModel(modelchoice='xception',
                             num_out_classes=num_out_classes)
    elif modelname == 'resnet18':
        return TransferModel(modelchoice='resnet18', dropout=dropout,
                             num_out_classes=num_out_classes)
    else:
        raise NotImplementedError(modelname)


def train_epoch(epoch, model, model_optim, loss, train_loader, device):
    train_loss = 0.0
    model.train()
    train_loss = []
    process = tqdm(train_loader, desc='train epoch %d ' % epoch)
    for data in process:
        img, label = data[0].to(device), data[1].to(device)
        model_optim.zero_grad()
        output = model(img)
        losses = loss(output, label)
        losses.backward()
        model_optim.step()
        train_loss.append(losses.item())
        process.set_description(desc='train epoch %d ' % epoch + 'loss :%0.5f' % (np.average(train_loss)))
    return np.average(train_loss)


def val_epoch(epoch, model, loss, val_loader, device):
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    model.eval()
    val_loss = []
    process = tqdm(val_loader, desc='val epoch %d ' % epoch)

    with torch.no_grad():
        for data in process:
            img, label = data[0].to(device), data[1].to(device)
            output = model(img)
            losses = loss(output, label)
            val_loss.append(losses.item())

            _, preds = torch.max(output, 1)
            val_correct += (preds == label).sum().item()
            val_total += label.size(0)

            process.set_description('val epoch %d ' % epoch + 'loss :%0.5f , acc:%0.2f' % (
                np.average(val_loss), 100 * val_correct / val_total))

    val_accuracy = 100 * val_correct / val_total
    print(f'Validation Accuracy: {val_accuracy:.2f}%')
    return val_accuracy


def train(model, epoch, save_path, model_optim, loss, device, train_loader, val_loader):
    train_losses = []
    val_acc = []
    best_acc = 0.0
    for epoch in range(epoch):
        train_loss = train_epoch(epoch, model, model_optim, loss, train_loader, device)
        val_accuracy = val_epoch(epoch, model, loss, val_loader, device)
        if val_accuracy > best_acc:
            torch.save(model, f'{save_path}/model_baseline2_test_test_{val_accuracy}.pt')
            print('Saving model...')
            best_acc = val_accuracy

        train_losses.append(train_loss)
        val_acc.append(val_accuracy)
    return train_losses, val_acc


def main():
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument('--batch_size', type=int, default=240, help='Batch size for training and validation')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the training on')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--train_txt', type=str, required=True, help='Path to the training dataset txt file')
    parser.add_argument('--val_txt', type=str, required=True, help='Path to the validation dataset txt file')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the Adam optimizer')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the training dataset')
    parser.add_argument('--val_path', type=str, required=True, help='Path to the validation dataset')
    args = parser.parse_args()

    train_dataset = MyDataset(args.train_txt, "train", args.train_path)
    val_dataset = MyDataset(args.val_txt, "val", args.val_path)
    all_dataset = ConcatDataset([train_dataset, val_dataset])

    train_dataloader = DataLoader(all_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)
    val_dataloader = DataLoader(all_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = model_selection('xception', num_out_classes=2).to(device)
    loss = nn.CrossEntropyLoss()
    model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)

    save_path = './checkpoint/'

    train_loss, val_acc = train(model, args.epochs, save_path, model_optim, loss, device, train_dataloader, val_dataloader)

if __name__ == '__main__':
    main()