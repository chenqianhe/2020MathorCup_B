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