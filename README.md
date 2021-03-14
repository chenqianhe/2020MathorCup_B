# 2020MathorCup第一届大数据赛B题解决方案

## 比赛简介

网址：https://www.saikr.com/contest/dm_detail_subB/41530

简而言之就是利用八张图片的训练集完成遥感**图像分割任务**，测试集为两张图片。

三个问题为：

问题1：计算10幅图中耕地在各图像中所占比例

问题2：从给定的2幅测试图像(Test1.tif、Test2.tif)中提取出耕地，制作耕地标签图，并将标签图分别上传到竞赛平台中（注意田块间的边界是否清晰）；

问题3：我国土地辽阔，地貌复杂，你有什么创新的思路能够快速、精准的识别出田块。

## 问题分析

问题一为面积占比计算问题。在耕地标签图的像素矩阵中，其中‘1’ 代表耕地类，‘0’代表背景类，我们只需要统计得到像素点为‘1’ 的数量即可求的耕地类的占比。

问题二为图像识别处理问题。从测试图像中提取出耕地，制作耕地标签图，要求我们在已有卫星遥感图像中将耕地识别出来，并与非耕地部分进行区分，其在图像的处理中具体体现为语义分割问题。而卫星遥感图像本身存在着**分辨率不够高**，**边缘模糊**等方面的问题，因此在制作标签图前我们可以使用**超分辨**的方法对图像进行处理，提高图像分辨率。如何对语义分割处理得到的图像进行优化也是我们需要思考的问题，综合田块间边界清晰，耕地连续完整性等现实因素考虑，我们需要对图像进行缺口连接、地块误判的修正、边缘平整等效果处理。

问题三就不是本文重点了，不过多论述。

## 下文结构

* 阈值操作
* 边缘提取
* 超分辨(RealSRPredictor)
* 语义分割(HRNet)
* 结果完善

## 阈值操作

阈的意思是界限，故阈值又叫临界值，是指一个效应能够产生的最低值或最高值。在该问题中，我们结合背景知识，考虑可以通过颜色特征进行划分，以 HSL 色域空间中的三参数阈值作为划分依据。RGB空间到HSL空间转换公式如下：

![](https://ai-studio-static-online.cdn.bcebos.com/c2492d14e7ed43cea6b1e4c3b410e60eb863380b52a147af955b7f8b23544bd5)

在进行相关测试后，根据现有的少量数据可以得知白色，紫色，青色，蓝色可以做 背景色，灰绿、红色、黄色可以做土地颜色，但是由于目前数据集的分布很不均匀（如图所示，此处均为以红色和绿色为阈值，即认为这两种颜色范围内为土地，阈值之外为背景，修改成黑色，可见同一个阈值对于不同图片的效果差异极大），导致这些特征缺乏说服力，同时无法利用仅有的八张图片做出一个很好的分类器和识别器。

![](https://ai-studio-static-online.cdn.bcebos.com/532c4c91343b4c23b9619ebb50e9b1eae5805e6404804e32a231085e7c814ddf)

相关代码如下：


```python
import cv2
result = cv2.imread()

(B, G, R) = cv2.split(result)
B = [i/255 for i in B]
G = [i/255 for i in G]
R = [i/255 for i in R]

for i in range(len(R)):
    for k in range(len(R[i])):
        max_ = max(R[i][k], G[i][k], B[i][k])
        min_ = min(R[i][k], G[i][k], B[i][k])
        h = 0.0
        s = 0.0
        l = 0.0

        if max_ == min_:
            h = 0;
        elif max_ == R[i][k] and G[i][k] > B[i][k]:
            h = 60*(G[i][k] - B[i][k])/(max_ - min_)
        elif max_ == R[i][k] and G[i][k] <= B[i][k]:
            h = 60*(G[i][k] - B[i][k])/(max_ - min_) + 360
        elif max_ == G[i][k]:
            h = 60*(B[i][k] - R[i][k])/(max_ - min_) + 120
        elif max_ == B[i][k]:
            h = 60*(R[i][k] - G[i][k])/(max_ - min_) + 240

        l = (max_ + min_)/2

        if max_ == min_ or l == 0:
            s = 0
        elif 0 < l <= 0.5:
            s = (max_ - min_) / (2 * l)
        elif l > 0.5:
            s =  (max_ - min_) / (2 - 2 * l)

```

## 边缘提取

边缘是图像最基本的特征，边缘提取对于图像目标的识别与分割具有重要意义。目 前常用的边缘检测方法是基于微分算子进行的，主要有 Roberts、Sobel、Prewitt、Canny 和 Log 等微分算子。Canny 和 Log 算子的检测精度较高，但算法较复杂，耗时较长，Roberts、 Sobel 和 Prewitt 算子的算法较为简单， 但存在着检测精度不高的问题。 相比之下， 基于 Sobel 的改进方法针对不同图像具有较好的可调性，进而可以获得较好的效果，但也存在着 边缘粗糙化的问题。在实际应用中我们选取了 Canny 进行了边缘提取的尝试。

![](https://ai-studio-static-online.cdn.bcebos.com/900d9006025f4508b2812da3e1781ba3af6c9d3f519c40d1a5bf0149086d84eb)

进行实际测试后发现，在测试集中 (如图所示)，由于土地之间也存在边缘，这种 细小的边缘也会被提取出来，导致得到的边缘非常杂乱而细密，但是我们又没有好的办法 进行有效边缘的筛选，导致该方法缺少适用性。

相关代码如下：


```python
import cv2
import matplotlib.pyplot as plt
def edge_demo(image):
    # GaussianBlur图像高斯平滑处理
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    # (3, 3)表示高斯矩阵的长与宽都是3,意思就是每个像素点按3*3的矩阵在周围取样求平均值，，标准差取0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   # 颜色模式转换成cv2.COLOR_BGR2GRAY模式下的灰度图像
   
    edge_output = cv2.Canny(gray, 50, 150)
    # 提取上一步中处理好的图像边缘，50和150分别代表低阈值和高阈值，高阈值用来将物体与背景区分开来，低的用于平滑连接高阈值产生的片段，使图像成一个整体
    plt.imshow(edge_output[:, :, [2,1,0]])
    # 原图与灰度图像与运算，按照灰度图剪切加和的原图
    dst = cv2.bitwise_and(image, image, mask=edge_output)

    plt.imshow(dst[:, :, [2,1,0]])


image = cv2.imread()
edge_demo(image)
```

## 语义分割

语义分割是计算机视觉中的基础任务，早期的语义分割有依赖聚类以及传统的机器 学习、统计学习方法。第一个采用神经网络进行语义分割的方法是 FCN(Fully Convolutional Networks) 全卷积神经网络，后来也出现了 SegNet、Unet、Deeplab、mask rcnn、Segfix 等。常 用于训练语义分割模型的数据集有 Pascal VOC 2012、Cityscapes、Pascal Context 和 Stanford Background Dataset。用于评估语义分割算法性能的标准指标是平均 IOU（Intersection Over Union，交并比），IoU 定义如下：

![](https://ai-studio-static-online.cdn.bcebos.com/ed816ad2d0c3460fb4a3b64d540043dea8184ecaaab6409aada19a2fec07559b)

IOU 数值越大，目标捕获程度越好，模型精确程度越高。 

U-Net 通过产生原始训练数据的扭曲版而增加训练数据。这一步使 CNN 编码器-解码器变得更加鲁棒以抵抗这些形变，并能从更少的训练图像中进行学习。DeepLab 将 CNN 编码器-解码器和 CRF 精炼过程相结合以产生目标标签。空洞卷积（也称扩张卷积）在每一 层都使用大小不同的卷积核，使每一层都能捕获各种比例的特征。

而本文采用的 HRNet 语义分割模型是近年来新提出的一种基于分层多尺度注意力的 语义分割模型。在 Cityscapes test 上可达 85.1% mIoU，性能优于 SegFix、HRNetV2-OCR 等网络。在减少内存的同时增加了预测的精确度，并改善了类别混淆的问题。

具体内容将在后文详细介绍。

## 超分辨模型

本文采用的 PaddlePaddle 预训练的 RealSR 基于模糊核和噪声注入的真实世界 超分辨率模型使用了一种新的基于核估计和噪声注入的退化框架。 通过不同的退化组合 (例如，模糊和噪声) 的使用，令获得的 LR 图像与原始图像具有相同的域属性。利用这些领 域一致的数据，用 patch discriminator 训练一个真实的超分辨率 GAN，从而达到更好的感知 HR 效果。实验表明，RealSR 优于最先进的方法，降低噪音的同时具有更好的视觉质量。


### 降级方法

RealSR 中用到的用到的降级方法主要有从真实数据中估计退化情况，并生成真实的 LR 图像和基于构建的数据对 SR 模型进行训练两部分。降级操作框架如图所示。

![](https://ai-studio-static-online.cdn.bcebos.com/f27982e9b49d4e77821ac43466e62a5bb8e1ad94334643578680dfc0a71342f9)

### 退化方法

RealSR 超分辨模型提出了一种基于核估计和噪声注入的真实图像退化方法。假设 LR 图像通过以下退化方法得到:

![](https://ai-studio-static-online.cdn.bcebos.com/657f9f31f7d047828ad165fb0f29bf92af2c238b476b440e89af24d6322fa107)

其中 k 和 n 分别表示模糊核和噪声，I_LR 和 (I_HR 分别表示 LR 和 HR 图像，s 为尺度 因子。I_HR 未知，则 k 和 n 也未知。

数据构造管道算法如下所示：

![](https://ai-studio-static-online.cdn.bcebos.com/b036859a238d475f8c00d699425dfce7ebf930d653db47f9b2826753b730c436)

其中, 核估计中估计的内核需要满足以下约束条件：

![](https://ai-studio-static-online.cdn.bcebos.com/ceabd4c7467147f4bf3a935d746998954ae59268b01c49a782a21e1e44c13119)

### 效果对比

我们将卫星遥感图像进行了分割 → 超分辨 → 重新拼合处理 → 再压缩，得到了分辨 率提高的图像。Test1.tif、Test2.tif 经超分辨处理后得到的标签图与原图像得到的标签图效 果对比如图所示。

![](https://ai-studio-static-online.cdn.bcebos.com/06d083127fee469b9c71b5d20a1fb094a8d2a7b4244640679b9480c198236fab)

从中可以明显看出经过 RealSR 超分辨处理后得到的耕地标签图相比于原始图片直接 得到的标签图具有边缘更清晰，识别度更高的特点，非耕地类面积有所增加，语义分割结果更为准确，因此超分辨模型对遥感图像地块的分割与提取具有重要意义。

### 相关代码


```python
!git clone https://gitee.com/paddlepaddle/PaddleGAN.git
%cd PaddleGAN/
!pip install -v -e .
```


```python
import os
from PaddleGAN.ppgan.apps.realsr_predictor import RealSRPredictor

sr = RealSRPredictor()

for i in os.listdir("small2"):
    sr.run("small2/"+i)
```

## 语义分割模型

为了精确的提取出耕地，制作耕地标签图，我们需要建立合适的语义分割模型。通 过查阅资料，我们选取 HRNet [3] 基于分层多尺度注意力的语义分割方法来对超分辨后的 图像进行处理，进行标签图的初步制作。利用 HRNet 预训练模型，我们提出了两种训练方 案，方案一为基于 Cityscapes 预训练得到 HRNet_W18，方案二为基于 Cityscapes 预训练和 CCFBDCI 遥感数据集迁移得到 HRNet_W18（基础模型均为 PaddlePaddle [1] 预训练），将 两者进行效果对比，选用了前一种方案。

### HRNet

#### 多尺度预测

多尺度预测常用来提高语义分割的结果。多种图像尺度通过经过网络，然后利用平均 池化或最大池化来得到。在这里我们将 attention 多层级机制与多尺度预测相结合，以达到 加快训练速度，提高训练精度的效果。基于注意力的方法组合多尺度预测，在 Cityscapes test 上可达 85.1% mIoU， 在 Mapillary val 上高达 61.1% mIoU， 表现真 SOTA， 性能优于 SegFix、HRNetV2-OCR 等网络。

#### 金字塔池化

在 HRNet 中我们运用到金字塔池化的原理。金字塔池化用于解决不同尺寸的输入图 片应用到已训练好的网络中去。它解决了输入图片大小不一造成的缺陷，具有从不同角度 提取图像特征再聚合的特点，显示了算法的鲁棒性，在目标识别中增加了精度，提高了模 型的精度。步骤示意图如图所示。

![](https://ai-studio-static-online.cdn.bcebos.com/dec8932efd8842c7b2668a75e6cc36033fb1a3962ec54e778e759f0d75659be2)

#### HRNet解决的问题

在做语义分割任务时，我们会发现对于有的图像在低尺度下可以取得更好效果，而有 的图像在高尺度上可以取得更好的效果。为了解决这一问题，HRNet 提出了两种方法：多 尺度注意机制和自动标记。HRNet 的多尺度注意机制可以忽视推理过程中的尺度数目，提 升识别效果的同时可视化的显示不同尺度和场景的重要性。每个尺度都学习一个稠密的掩模，这些多尺度预测通过在掩模之间执行像素相乘与预测相结合，然后在不同尺度之间进 行像素累加，以获得最终结果。粗糙的标签有噪音和不准确性。HRNet 使用自动标记作为 产生更丰富标签的手段，从而填补标签的空白，通过填补长尾数据分布的空白，有助于泛 化。

#### 数据增强

深层神经网络一般都需要大量的训练数据才能获得比较理想的结果。在数据量有限 的情况下，可以通过数据增强（Data Augmentation）来增加训练样本的多样性，提高模型 鲁棒性，避免过拟合。数据增强的另一种解释是，随机改变训练样本可以降低模型对某些 属性的依赖，从而提高模型的泛化能力。我们通过对图像进行随机水平翻转、随机上下翻 转、随机旋转、随机图像截取，让物体以不同的比例出现在图像的不同位置，降低模型对 目标位置的敏感性。通过设置随机亮度、随机对比度、随机饱和度、随机色调来降低模型 对色彩的敏感度。增加了样本的多样性，提高了模型的泛化能力。

#### 相关代码



```python
!pip install paddlex
```


```python
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

```


```python
import paddlex as pdx

print("Loading model...")
model = pdx.deploy.Predictor('inference_model', use_gpu=True)
print("Model loaded.")


for i in []:
    result = model.predict(i)
    pdx.seg.visualize(i, result, weight=0.0, save_dir='./')
```

### 两种模型效果对比

我们将基于 Cityscapes 预训练得到的 HRNet_W18 与基于 Cityscapes 预训练和 CCFBDCI 遥感数据集迁移得到的 HRNet_W18 进行效果对比来决定最终方案。本文使用 VisualDL 的可视化功能来辅助调整参数，通过 VisualDL-Scalar 查看训练过程，通过观察训练参数变 化，了解训练过程，加速模型调优。我们先对训练方案一进行可视化处理，基于 Cityscapes 预训练得到的 HRNet_W18 调整过程如图所示。

![](https://ai-studio-static-online.cdn.bcebos.com/a137c29996014daf8592b3693cf078041646fba3010f46eda75ef4220c0bb40c)

我们对 HRNet 语义分割模型预训练方案一进行了模型评估， 选择合适的增强策略， 得到最优的学习率，本文将训练参数中迭代轮数设为 100，学习率设为 0.015。模型评估如下所示。

![](https://ai-studio-static-online.cdn.bcebos.com/86214628ef18420d99b7867423194839dd3b73cba13546eb94edd2a9765b228b)

基于 Cityscapes 预训练得到的 HRNet_W18 的平均交并比（mIoU）达到了 80.90%，平 均准确率（mAcc）达到了 92.93%，达到了较好的训练效果。

用同样的辅助调整参数方法来调整基于 Cityscapes 预训练和 CCFBDCI 遥感数据集迁 移得到的 HRNet_W18，得到最优的学习率并进行模型评估，模型的平均交并比如图示。

![](https://ai-studio-static-online.cdn.bcebos.com/707947d3e5824381a5752aac567ebae18a9150410af944c7b85e1899b3f07f8f)

由图可知，方案二模型的平均交并比在 24.00% 到 30.00% 之间，远小于方案一模型的 平均交并比，可能的原因为 CCFBDCI 遥感数据集对地块的种类划分中不存在耕地类，在后 续的深度学习过程中导致了对耕地类地块的识别效果不佳。因此我们选用基于 Cityscapes 预训练得到的 HRNet_W18 来作为我们的训练模型。

### 模型效果检验

使用训练好的 HRNet 语义分割模型制作 2 幅测试图像 (Test1.tif、Test2.tif) 经 RealSR 超分辨处理后的初始耕地标签图如图所示。

![](https://ai-studio-static-online.cdn.bcebos.com/71fa16774c4145138d72e799db0bad83f301cdaa60df496a9f6af27f39d4aa60)

我们可以看到两幅预测图像已经基本完成了遥感图像地块的划分与提取，但从图中 也可观察到由 HRNet 预测得到的耕地标签图还存在着线段不连续，地块不完整，边缘粗糙 等问题，地块分割与提取的精确度还不够，因此我们需要设定一些条件函数来使模型结果 满足我们的需求。

## 优化处理

### 优化流程

本文进行的基于超分辨和 HRNet 的遥感图像的地块分割与提取具体流程如图所示，在利用 RealSR 超分辨模型和 HRNet 语义分割模型处理遥感图像后，我们还进行了一 系列优化处理操作，以达到更好的分割与提取效果。

![](https://ai-studio-static-online.cdn.bcebos.com/7f96ab1646454a35b7bb8f9c7e000f02022ae827ccfb4c25be61ba0a430ecde5)

### 主要思路

由于图中耕地面积占比较大，由面积差值所引起的相对误差较非耕地小，我们在部分 叠加过程中会将非耕地类作为优先级考虑。我们首先将卫星遥感图像直接应用训练好的模 型处理，得到结果一，为后续步骤的叠加提供材料。再重新对卫星遥感图像进行超分辨处 理，提高图像分辨率，使其达到更好的预测结果，经 HRNet 模型处理后得到结果二。为优 化结果二的分割效果，我们对其进行边缘连接得到线段连接图结果三，将结果二和结果三 直接叠加，得到结果四，达到边缘清晰平整的效果。为修正地块的误判，我们进行从小到 大的三次去误差处理。首先进行 3×3 去误差处理得到结果五，综合考虑非耕地为优先级， 将结果二和结果五的非耕地部分叠加得到结果六，经 5×5 去误差处理后得到结果七，为减 小优化处理过程中可能出现的地块丢失的影响，将结果一与结果七直接叠加得到结果八， 经 9×9 去误差处理后得到最终标签图。

### 边缘连接

从图 6-4初始标签图中我们可以看到存在着耕地类边缘粗糙化，地块分割线不连续的 问题， 因此对标签图进行边缘连接处理具有重要优化价值。 在实际编程过程中， 我们用 sobel 算子得到梯度幅度和梯度角度阵列，进行行扫描，间隔 k 时进行填充，填充值为 ‘1’， 选取边缘，提取边缘坐标，将相应坐标值设为 ‘1’，再进行水平、垂直边缘连接。

### 去误差操作

从图 6-4初始标签图中我们可以发现存在着大块耕地中非耕地的误判以及大块非耕地 中耕地的误判现象。在实际编程过程中，我们使用滑块来对误判进行检测，修正大区域中 极小区域的误判部分，误判的判定条件为非同类数目过半。为便于确定中心像素块，我们 使用“奇数 × 奇数”的滑块模型来进行我们的操作。

针对小块误差，由于滑块区域范围过小会导致误差地块占比过大，检测正确率较低， 因此滑块区域选择在合适范围内越大，小块误差被检测出来的概率越大。而对于稍大块误 差，滑块区域范围较小时作用较好，检验正确率较高。为了对误判进行合理有效的修复， 我们从小范围到大范围进行三次去误差处理。经实际情况验证，这种去误差处理方式是有 效的。例如 5×5 滑块检测，以检测目标像素块为中心选取一个 5×5 的滑块，若除去中心的 24 个像素块中有超过半数的像素块标签值与中心像素块的标签值不同，则判定检测目标像 素块为误判对象，并修正其标签值。通过滑块的移动来实现对整张标签图的误判修正。操 作示意图如图所示。

![](https://ai-studio-static-online.cdn.bcebos.com/1050a5baeab84c32a48793a8bd48f700c21ae7f6f5de411b8e6670b4dfec5739)

### 优化处理效果图

![](https://ai-studio-static-online.cdn.bcebos.com/414831ae3dc547fda4f2808cc22cd9cdfed693a9891345119f761fbb01b2c83d)

从图中可以看出，结果四经边缘连接后标签图边缘平整度和线段连续度得到了较大 的提升，最终结果经去三次误差处理后地块的完整程度明显提高，对误判的修复效果较好。 最终标签图已经很好的满足了题目要求，证明我们的模型优化处理效果较好且对于标签图 的制作具有重要价值。

### 相关代码


```python
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_pics(path):
    '''
    灰度读取图片
    并将非1转化为1
    返回图片数组
    '''
    #读取图像，支持 bmp、jpg、png、tiff 等常用格式
    #第二个参数是通道数和位深的参数，有四种选择，参考https://www.cnblogs.com/goushibao/p/6671079.html

    # IMREAD_UNCHANGED = -1#不进行转化，比如保存为了16位的图片，读取出来仍然为16位。
    # IMREAD_GRAYSCALE = 0#进行转化为灰度图，比如保存为了16位的图片，读取出来为8位，类型为CV_8UC1。
    # IMREAD_COLOR = 1#进行转化为RGB三通道图像，图像深度转为8位
    # IMREAD_ANYDEPTH = 2#保持图像深度不变，进行转化为灰度图。
    # IMREAD_ANYCOLOR = 4#若图像通道数小于等于3，则保持原通道数不变；若通道数大于3则只取取前三个通道。图像深度转为8位
    img = cv2.imread(path,2)
    img[img==14] = 1
    print(img)

    print(img.shape)
    print(img.dtype)
    print(img.min())
    print(img.max()) 
    plt.imshow(img)
    return img


def edge_linking(path):
    '''
    水平和竖直方向连接间距超过k的线段
    '''
    img2 = cv2.imread(path)
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray = gray / 255.0 #像素值0-1之间

    #sobel算子分别求出gx，gy
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=1) #得到梯度幅度和梯度角度阵列
    g = np.zeros(gray.shape) #g与图片大小相同

    #行扫描，间隔k时，进行填充，填充值为1
    def edge_connection(img, size, k):
        for i in range(size):
            Yi = np.where(img[i, :] > 0)
            if len(Yi[0]) >= 10: #可调整
                for j in range(0, len(Yi[0])-1):
                    if Yi[0][j+1] - Yi[0][j] <= k:
                        img[i, Yi[0][j]:Yi[0][j+1]] = 1
        return img

    #选取边缘，提取边缘坐标，将g中相应坐标像素值设为1
    X, Y = np.where((mag > np.max(mag) * 0.3)&(ang >= 0)&(ang <= 90))
    g[X, Y] = 1

    #边缘连接，此过程只涉及水平，垂直边缘连接，不同角度边缘只需旋转相应角度即可
    g = edge_connection(g, gray.shape[0], k=10)
    g = cv2.rotate(g, 0)
    g = edge_connection(g, gray.shape[1], k=10)
    g = cv2.rotate(g, 2)

    for i in range(500):
        for j in range(600):
            if g[i][j]:
                g[i][j] = 0
            else:
                g[i][j] = 1

    plt.imshow(g)
    return g
    

def superposition1(img, g):
    '''
    得到的线段连接图和原图进行叠加
    '''
    for i in range(500):
        for j in range(600):
            g[i][j] += img[i][j]
    plt.imshow(g)
    for i in range(500):
        for j in range(600):
            if g[i][j]==2:
                g[i][j] = 1
            else:
                g[i][j] = 0
    plt.imshow(g)
    return g
    
    
def sliding_treatment2(g):
    '''
    5*5的滑块进行检测，修整大区域中极小区域的误判部分
    '''
    for i in range(2,498):
        for j in range(2,598):
            flag = 0
            for k in range(-2,3):
                for l in range(-2,3):
                    if g[i+k][j+l] != g[i][j]:
                        flag += 1
            if flag > 12:
                if g[i][j]:
                    g[i][j] = 0
                else:
                    g[i][j] = 1

    plt.imshow(g)
    return g
    
    
def superposition2(img, g):
    '''
    和原图进行叠加，填补非土地部分
    '''
    for i in range(0,500):
        for j in range(0,600):
            if g[i][j] != img[i][j] and img[i][j]==0:
                g[i][j] = img[i][j]
    plt.imshow(g)
    return g
    
    
def sliding_treatment1(g):
    '''
    3*3的滑块进行检测，修整大区域中极小区域的误判部分
    '''
    for i in range(1,499):
        for j in range(1,599):
            flag = 0
            for k in range(-1,2):
                for l in range(-1,2):
                    if g[i+k][j+l] != g[i][j]:
                        flag += 1
            if flag > 4:
                if g[i][j]:
                    g[i][j] = 0
                else:
                    g[i][j] = 1

    plt.imshow(g)
    return g


def sliding_treatment3(g):
    '''
    9*9的滑块进行检测，修整大区域中极小区域的误判部分
    '''
    for i in range(4,496):
        for j in range(4,596):
            flag = 0
            for k in range(-4,5):
                for l in range(-4,5):
                    if g[i+k][j+l] != g[i][j]:
                        flag += 1
            if flag > 9*9//2:
                if not g[i][j]:
                    g[i][j] = 1

    plt.imshow(g)
    return g
    
    
if "__name__"=="main":
    img = load_pics("visualize_Test1_super_resolution.png")
    img2 = load_pics("visualize_Test1.png")
    g = edge_linking("visualize_Test1_super_resolution.png")
    
    g = superposition1(img, g)
    
    g = sliding_treatment1(g)
    
    g = superposition2(img, g)
    
    g = sliding_treatment2(g)
    
    g = superposition1(img2, g)
    
    g = sliding_treatment3(g)
    
    G = Image.fromarray(np.array(g))
    if G.mode == "F":
        G = G.convert("L") 
    G.save("result/Test1_reference.tif")
```

## 最终体系结构

* **RealSR** 超分辨针对现有数据进行超分辨处理，获得更加清晰的图像数据，使得后续 处理效果整体提升。

* **HRNet** 语义分割利用数据增强的方法在 PaddlePaddle 预训练 HRNet 上进行迁移学习， 得到效果较好的语义分割模型，处理图片得到初步标签图。

* **结果优化** 使用边缘连接、滑块检测等手段，达到了边缘的平整化清晰化，实现了误判 的修复，进一步处理图片得到了最终的标签图。

## 提升思路

首先，针对 HRNet 模型我们未使用类别权重这一超参数，由于土地和背景两类别的 占比很不均衡，因此有理由猜想当使用权重后我们可以得到更好的结果；除此之外，我们 可以在模型迁移上采取一些更好的方式，例如利用上 CCF 遥感数据集，当我们对 CCF 的 数据集标注进行合并修改后，即背景为一类，非背景合并为一类，这样一来是可以更快的训练并得到更好的结果。 

其次，我们也应该考虑结合传统方法做更多的工作，针对色彩这一个特征来说，土地 和背景其实有较为明显的特征区别，因此在有更大的数据之后，我们可以利用色彩这个特 征得到更加合适的分类器和滤波器；同时可以将色彩筛选和边缘提取进行结合，综合提升 效果。最重要的是可以将传统方法和深度学习方法进行结合，互相补充以得到最为合适的 结果。

第三点，我们在去误差处理时，是直接统计不同像素点的个数是否过半，这种方法不 一定是最优的，因此我们可以构建一个神经网络，来学习相关参数，以获得更好的效果。
