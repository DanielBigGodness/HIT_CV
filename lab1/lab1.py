import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


# dirname = os.path.dirname(PySide2.__file__)
# plugin_path = os.path.join(dirname, 'plugins', 'platforms')
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path


def image_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#要求一：SIFT提取特征点
def sift(image):
    descriptor = cv2.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)
    kps = np.float32([kp.pt for kp in kps])
    return kps, features


#要求二：特征点匹配
# 定义一个函数，根据两组关键点和特征，找出最佳的特征匹配对和变换矩阵
def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
    # 创建一个暴力匹配器对象
    matcher = cv2.BFMatcher()
    # 使用knn方法，对两组特征进行匹配，返回最近的两个匹配结果
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

    # 定义一个空列表，存储最佳的特征匹配对
    best_matches = []
    # 定义一个变量，存储最大的匹配对数量
    max_matches_num = 0
    # 定义一个变量，存储期望的匹配距离
    ex_dist = 100
    # 定义一个变量，存储匹配距离的容差
    derta_dist = 10
    # 循环10次，尝试找出最佳的匹配对
    for i in range(10):
        # 定义一个空列表，存储当前的匹配对
        matches = []
        # 定义一个空列表，存储当前的匹配距离
        dist_list = []
        # 遍历原始的匹配结果
        for m in rawMatches:
            # 如果匹配结果有两个，并且第一个匹配的距离小于第二个匹配的距离乘以一个比例
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                # 如果第一个匹配的距离在期望的距离的容差范围内
                if ex_dist - derta_dist <= m[0].distance <= ex_dist + derta_dist:
                    # 将第一个匹配的索引添加到当前的匹配对列表中
                    matches.append((m[0].trainIdx, m[0].queryIdx))
                    # 将第一个匹配的距离添加到当前的匹配距离列表中
                    dist_list.append(m[0].distance)
        # 从当前的匹配距离列表中随机选择一个作为新的期望的匹配距离
        ex_dist = dist_list[random.randint(0, len(dist_list) - 1)]
        # 如果当前的匹配对数量大于等于最大的匹配对数量
        if max_matches_num <= len(matches):
            # 更新最大的匹配对数量
            max_matches_num = len(matches)
            # 更新最佳的匹配对列表
            best_matches = matches

    # 如果最佳的匹配对数量大于4
    if len(best_matches) > 4:
        # 根据最佳的匹配对，获取两组关键点的坐标
        ptsA = np.float32([kpsA[i] for (_, i) in best_matches])
        ptsB = np.float32([kpsB[i] for (i, _) in best_matches])
        #要求三
        # 使用RANSAC方法，根据两组关键点的坐标，计算变换矩阵和状态
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        # 返回最佳的匹配对，变换矩阵和状态
        return best_matches, H, status
    # 如果最佳的匹配对数量小于等于4，返回None
    return None


#要求五：根据变换拼接
def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
    # 初始化可视化图片，将A、B图左右连接到一起
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    ptAs = []
    ptBs = []
    # 联合遍历，画出匹配对
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # 当点对匹配成功时，画到可视化图上
        if s == 1:
            # 画出匹配对
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
            ptAs.append(ptA)
            ptBs.append(ptB)

    # 返回可视化结果
    return vis, ptAs, ptBs


file_path = "./images/multiple_stitching"
dirs = os.listdir(file_path)
images = []
# image_stitched = []
for file in dirs:
    image = file_path + '/' + file
    image = cv2.imread(image)
    images.append(image)


def stitch(images, ratio=0.95, reprojThresh=4.0):
    # 获取输入图片
    (imageA, imageB) = images
    top, bottom, left, right = 400, 400, 400, 400
    imageA = cv2.copyMakeBorder(imageA, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    imageB = cv2.copyMakeBorder(imageB, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # 检测A、B图片的SIFT关键特征点，并计算特征描述子
    (kpsA, featuresA) = sift(imageA)
    (kpsB, featuresB) = sift(imageB)

    # 匹配两张图片的所有特征点，返回匹配结果
    M = matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

    # 如果返回结果为空，没有匹配成功的特征点，退出算法
    if M is None:
        return None

    # 否则，果提取匹配结
    # H是3x3视角变换矩阵
    (matches, H, status) = M
    result = cv2.warpPerspective(imageA, H, (imageB.shape[1], imageB.shape[0])) #透视变换

    #内点更新
    for i in range(imageA.shape[0]):
        for j in range(imageA.shape[1]):
            if imageA[i, j][0] >= result[i, j][0]:
                result[i, j] = imageB[i, j]

    # 检测是否需要显示图片匹配
    (imageA, imageB) = images
    (kpsA, featuresA) = sift(imageA)
    (kpsB, featuresB) = sift(imageB)
    M = matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
    (matches, H, status) = M
    vis, ptA, ptB = drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
    # 返回结果
    return result, vis, ptA, ptB


def cv2_show_plt(img):
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    return img


imageA = images[0]
imageB = images[1]
imageC = images[2]
(result_ab, vis_ab, pointab, pointba) = stitch([imageA, imageB])
(result_bc, vis_bc, pointcb, pointbc) = stitch([imageC, imageB])
for point in pointab:
    x, y = int(point[0]), int(point[1])
    cv2.circle(vis_ab, (x, y), 5, (255, 0, 0), -1)
for point in pointba:
    x, y = int(point[0]), int(point[1])
    cv2.circle(vis_ab, (x, y), 5, (0, 0, 255), -1)
for point in pointbc:
    x, y = int(point[0]), int(point[1])
    cv2.circle(vis_bc, (x, y), 5, (255, 0, 0), -1)
for point in pointcb:
    x, y = int(point[0]), int(point[1])
    cv2.circle(vis_bc, (x, y), 5, (0, 0, 255), -1)
result = result_ab
for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        if result_bc[i, j][0] >= result[i, j][0]:
            result[i, j] = result_bc[i, j]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2_show_plt(vis_ab))
plt.title("vis_ab")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(cv2_show_plt(result_ab))
plt.title("result_ab")
plt.axis("off")
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2_show_plt(vis_bc))
plt.title("vis_bc")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(cv2_show_plt(result_bc))
plt.title("result_bc")
plt.axis("off")
plt.show()

plt.figure(figsize=(5, 5))
plt.axis("off")
plt.imshow(cv2_show_plt(result))
plt.title("result")
plt.show()
