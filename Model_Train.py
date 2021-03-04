import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import matplotlib
matplotlib.use('Agg')

import paddlex as pdx
from paddlex.seg import transforms

# 定义训练和验证时的transforms
train_transforms = transforms.Compose([
    transforms.RandomPaddingCrop(crop_size=769),
    transforms.RandomHorizontalFlip(prob=0.5), 
    transforms.RandomVerticalFlip(prob=0.5),
    transforms.RandomBlur(prob=0.5),
    transforms.RandomRotate(rotate_range=35),
    transforms.RandomDistort(brightness_prob=0.5, contrast_prob=0.5, saturation_prob=0.5, hue_prob=0.5),
    transforms.Normalize()
])

eval_transforms = transforms.Compose(
    [transforms.Padding(target_size=769), transforms.Normalize()])
    

#定义数据集
train_dataset = pdx.datasets.SegDataset(
    data_dir='dataset',
    file_list='dataset/train_list.txt',
    label_list='dataset/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.SegDataset(
    data_dir='dataset',
    file_list='dataset/val_list.txt',
    label_list='dataset/labels.txt',
    transforms=eval_transforms)
    

#模型训练
num_classes = len(train_dataset.labels)

model = pdx.seg.HRNet(num_classes=num_classes, width=18, use_bce_loss=False, use_dice_loss=False, class_weight=None, ignore_index=255, input_channel=3)

model.train(
    num_epochs=100,
    train_dataset=train_dataset,
    train_batch_size=3,
    eval_dataset=eval_dataset,
    learning_rate=0.015,
    save_interval_epochs=10,
    pretrain_weights='CITYSCAPES',
    save_dir='output2/hrnet', #可视化结果保存在该目录的 vdl_log 文件夹下
    use_vdl=True) #使用内置的VisualDL