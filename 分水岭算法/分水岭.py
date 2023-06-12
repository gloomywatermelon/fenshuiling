import cv2
import numpy as np
from matplotlib import pyplot as plt

# 加载图像
image = cv2.imread('image.jpg')

# 灰度化处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 进行阈值分割
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 创建一个卷积核
kernel = np.ones((3, 3), np.uint8)

# 开运算，用于去除噪音
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# 膨胀操作，增加前景区域
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# 距离变换
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

# 阈值化
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# 找到不确定区域
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# 标记连通区域
_, markers = cv2.connectedComponents(sure_fg)

# 增加标记，将不确定区域标记为0
markers = markers + 1
markers[unknown == 255] = 0

# 应用分水岭算法
markers = cv2.watershed(image, markers)

# 将分水岭区域标记为红色
image[markers == -1] = [0, 0, 255]

# 保存结果
cv2.imwrite('output_image.jpg', image)

# 显示原始图像和结果
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Input Image')
plt.subplot(122), plt.imshow(cv2.cvtColor(cv2.imread('output_image.jpg'), cv2.COLOR_BGR2RGB)), plt.title('Output Image')
plt.show()
