import os
import glob

import albumentations
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import models

from src.modeling.datasets.image_path_dataset import ImagePathDataset


DEVICE = 'cuda:0'
PATCH_SIZE = 128
TRAIN_BATCH_SIZE = 1
TEST_BATCH_SIZE = 64
EPOCHS = 100


def main():
    dataset_path = r'C:\data\mvtec\bottle\train\good'
    experiment_path = r'C:\dev\anomaly_detection\models'
    im_paths = glob.glob(os.path.join(dataset_path, '**/*png'), recursive=True)
    print('[INFO] found {} images to train on'.format(len(im_paths)))
    # create data loaders
    train_augs = [albumentations.HorizontalFlip(p=0.5),
                  albumentations.RandomCrop(height=PATCH_SIZE, width=PATCH_SIZE, always_apply=True, p=1),
                  albumentations.Normalize(always_apply=True, p=1),
                  ToTensorV2()]
    train_augs = albumentations.Compose(train_augs)
    train_dataset = ImagePathDataset(im_paths=im_paths, transforms=train_augs)
    train_loader = DataLoader(dataset=train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

    # create the networks
    pretrained_vgg = models.vgg19(pretrained=True)
    teacher = pretrained_vgg.features[:36]
    teacher = teacher.to(DEVICE)

    vgg = models.vgg19(pretrained=False)
    student = vgg.features[:36]
    student = student.to(DEVICE)

    # train the model
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(student.parameters(), lr=0.0002, weight_decay=0.00001)
    teacher.eval()

    for epoch in tqdm(range(EPOCHS)):
        for img in train_loader:
            img = img.to(DEVICE)
            with torch.no_grad():
                surrogate_label = teacher(img)
            optimizer.zero_grad()
            pred = student(img)

            loss = criterion(pred, surrogate_label)
            loss.backward()
            optimizer.step()

    torch.save(teacher.state_dict(), os.path.join(experiment_path, 'teacher.pth'))
    torch.save(student.state_dict(), os.path.join(experiment_path, 'student.pth'))


if __name__ == '__main__':
    main()
