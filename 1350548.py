
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def Calculation(path):
    img = cv2.imread(path,2)
    count = 0
    all = 500 * 600
    for i in range(500):
        for j in range(600):
            if img[i][j]:
                count += 1
    return count/all
    
rate = []
Path = ['Data1_reference.tif',
 'Data2_reference.tif',
 'Data3_reference.tif',
 'Data4_reference.tif',
 'Data5_reference.tif',
 'Data6_reference.tif',
 'Data7_reference.tif',
 'Data8_reference.tif',
 'Test1_reference.tif',
 'Test2_reference.tif']
for i in Path:
    rate.append(Calculation("result/"+i))
print(rate)

get_ipython().run_line_magic('matplotlib', 'inline')
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


get_ipython().run_line_magic('matplotlib', 'inline')

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


def sliding_treatment1(g):
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


def sliding_treatment2(g):
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


img = load_pics("visualize_Test1_super_resolution.png")
img2 = load_pics("visualize_Test1.png")
g = edge_linking("visualize_Test1_super_resolution.png")
g = superposition1(img, g)
g = sliding_treatment1(g)
g = superposition2(img, g)
g = sliding_treatment1(g)
g = superposition1(img2, g)
plt.imshow(g)


