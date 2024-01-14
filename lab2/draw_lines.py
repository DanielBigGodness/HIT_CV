import numpy as np
import cv2


def draw_lines(image, lines, color=[0, 255, 0], thickness=2):

    right_y_set = []
    right_x_set = []
    right_slope_set = []

    left_y_set = []
    left_x_set = []
    left_slope_set = []

    slope_min = .35  # 斜率低阈值
    slope_max = .85  # 斜率高阈值
    middle_x = image.shape[1] / 2  # 图像中线x坐标
    max_y = image.shape[0]  # 最大y坐标
    min_y = int(max_y * 0.6)

    for line in lines:
        x1 = line[0][0]
        y1 = line[0][1]
        x2 = line[1][0]
        y2 = line[1][1]
        fit = np.polyfit((x1, x2), (y1, y2), 1)    # 拟合成直线
        slope = fit[0]  # 斜率

        if slope_min < np.absolute(slope) <= slope_max:

            # 将斜率大于0且线段X坐标在图像中线右边的点存为右边车道线
            if slope > 0:
                right_y_set.append(y1)
                right_y_set.append(y2)
                right_x_set.append(x1)
                right_x_set.append(x2)
                right_slope_set.append(slope)

            # 将斜率小于0且线段X坐标在图像中线左边的点存为左边车道线
            elif slope < 0:
                left_y_set.append(y1)
                left_y_set.append(y2)
                left_x_set.append(x1)
                left_x_set.append(x2)
                left_slope_set.append(slope)

    # 绘制左车道线
    if left_y_set:
        lindex = left_y_set.index(min(left_y_set))  # 最高点
        left_x_top = left_x_set[lindex]
        left_y_top = left_y_set[lindex]
        lslope = np.median(left_slope_set)   # 计算平均值

        # 根据斜率计算车道线与图片下方交点作为起点
        left_x_bottom = int(left_x_top + (max_y - left_y_top) / lslope)

        # 根据斜率计算车道线顶部终点
        left_x_top = int(left_x_top + (min_y - left_y_top) / lslope)

        # 绘制线段
        cv2.line(image, (left_x_bottom, max_y), (left_x_top, min_y), color, thickness)

    # 绘制右车道线
    if right_y_set:
        rindex = right_y_set.index(min(right_y_set))  # 最高点
        right_x_top = right_x_set[rindex]
        right_y_top = right_y_set[rindex]
        rslope = np.median(right_slope_set)

        # 根据斜率计算车道线与图片下方交点作为起点
        right_x_bottom = int(right_x_top + (max_y - right_y_top) / rslope)

        right_x_top = int(right_x_top + (min_y - right_y_top) / rslope)

        # 绘制线段
        cv2.line(image, (right_x_top, min_y), (right_x_bottom, max_y), color, thickness)
