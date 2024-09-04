import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import timm
import yaml
import pandas as pd
import os
import argparse
from PIL import Image
import numpy as np
from FFDV_data import data_processing

def predict(test_loader, model, tta=10):
    # switch to evaluate mode
    model.eval()

    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            for i, input in enumerate(test_loader):
                input = input.cuda()

                # compute output
                output = model(input)
                output = F.softmax(output, dim=1)
                output = output.data.cpu().numpy()

                test_pred.append(output)
        test_pred = np.vstack(test_pred)

        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta


class FFDIDataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = img_path

        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.img_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing configuration")
    parser.add_argument('--config_path', type=str, required=True, help='Path to the configuration YAML file')
    args = parser.parse_args()

    with open(args.config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    video_input_path = config['video_input_path']
    output_dir = config['spectrum_output_dir']
    test_label_path = config['test_label_path']
    pretrained_weight_path = config['pretrained_weight_path']
    checkpoint_path = config['checkpoint_path']
    result_path = config['result_path']
    unknown = config['unknown']


    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    data_processing(video_input_path, output_dir)

    test_label = pd.read_csv(test_label_path)
    test_label['path'] = output_dir + test_label['video_name'].apply(
        lambda x: x[:-4] + '.jpg')

    test_label = test_label[test_label['path'].apply(os.path.exists)]

    test_loader = torch.utils.data.DataLoader(
        FFDIDataset(test_label['path'].values,
                    transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5] * 3, [0.5] * 3)
                        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    ), batch_size=256, shuffle=False, num_workers=6, pin_memory=True
    )

    #model = timm.create_model('resnet101', pretrained=True, num_classes=2,
                             # pretrained_cfg_overlay=dict(file=pretrained_weight_path))
    model = timm.create_model('resnet101', pretrained=False, num_classes=2)
    #model.load_state_dict(torch.load(pretrained_weight_path))

    model = model.cuda()
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(checkpoint_path), strict=True)

    test_label["y_pred"] = predict(test_loader, model, 1)[:, 1]
    submit = pd.read_csv(unknown)
    merged_df = submit.merge(test_label[['video_name', 'y_pred']], on='video_name', suffixes=('', '_df2'), how='left', )
    merged_df['y_pred'] = merged_df['y_pred_df2'].combine_first(merged_df['y_pred'])
    merged_df[['video_name', 'y_pred']].to_csv(result_path, index=None)
