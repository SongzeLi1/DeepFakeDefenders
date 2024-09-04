import argparse
import torch
import yaml
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset, ConcatDataset
import timm
import time
from FFDV_data import data_processing
import pandas as pd
import os
from PIL import Image
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = ""

    def pr2int(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc = (output.argmax(1).view(-1) == target.float().view(-1)).float().mean() * 100
            losses.update(loss.item(), input.size(0))
            top1.update(acc, input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f}'
              .format(top1=top1))
        return top1


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, losses, top1)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        acc = (output.argmax(1).view(-1) == target.float().view(-1)).float().mean() * 100
        top1.update(acc, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            progress.pr2int(i)


class FFDIDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label

        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.from_numpy(np.array(self.img_label[index]))

    def __len__(self):
        return len(self.img_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument('--config_path', type=str, required=True, help='Path to the configuration YAML file')
    args = parser.parse_args()

    with open(args.config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    train_video_input_path = config['train_video_input_path']
    train_output_dir = config['train_output_dir']
    train_label_path = config['train_label_path']
    val_video_input_path = config['val_video_input_path']
    val_output_dir = config['val_output_dir']
    val_label_path = config['val_label_path']
    pretrained_weight_path = config['pretrained_weight_path']
    checkpoint_output_path = config['checkpoint_output_path']

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    train_label = pd.read_csv(train_label_path)
    val_label = pd.read_csv(val_label_path)

    data_processing(train_video_input_path, train_output_dir)
    data_processing(val_video_input_path, val_output_dir)

    train_label['path'] = train_output_dir + train_label['video_name'].apply(
        lambda x: x[:-4] + '.jpg')
    val_label['path'] = val_output_dir + val_label['video_name'].apply(
        lambda x: x[:-4] + '.jpg')

    train_label = train_label[train_label['path'].apply(os.path.exists)]
    val_label = val_label[val_label['path'].apply(os.path.exists)]

    train_dataset = FFDIDataset(train_label['path'].values, train_label['target'].values,
                                transforms.Compose([
                                    transforms.Resize((256, 256)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5] * 3, [0.5] * 3)
                                ])
                                )

    val_dataset = FFDIDataset(val_label['path'].values, val_label['target'].values,
                              transforms.Compose([
                                  transforms.Resize((256, 256)),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.5] * 3, [0.5] * 3)
                              ])
                              )

    all_dataset = ConcatDataset([train_dataset, val_dataset])


    train_loader = torch.utils.data.DataLoader(
        all_dataset, batch_size=256, shuffle=True, num_workers=6, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        all_dataset, batch_size=256, shuffle=False, num_workers=6, pin_memory=True
    )


    #model = timm.create_model('resnet101', pretrained=True, num_classes=2,
                              #pretrained_cfg_overlay=dict(file=pretrained_weight_path))
    model = timm.create_model('resnet101', pretrained=False, num_classes=4)
    model = model.cuda()
    model = nn.DataParallel(model).cuda()
    # model.load_state_dict(torch.load('./checkpoint_2_phase2/model_use_all_DA_99.99907.pt'), strict=True)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), 0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.85)
    best_acc = 0.0
    for epoch in range(50):
        scheduler.step()
        print('Epoch: ', epoch)

        train(train_loader, model, criterion, optimizer, epoch)
        val_acc = validate(val_loader, model, criterion)

        if val_acc.avg.item() > best_acc:
            best_acc = round(val_acc.avg.item(), 5)
            torch.save(model.state_dict(), os.path.join(checkpoint_output_path, f'model_use_all_DA_{best_acc}.pt'))
