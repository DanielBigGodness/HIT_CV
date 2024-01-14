import numpy as np


def Hough_Line(edge, img):
    def voting(edge):
        H, W = edge.shape

        drho = 1
        dtheta = 1

        # get rho max length
        rho_max = np.ceil(np.sqrt(H ** 2 + W ** 2)).astype(int)

        # hough table
        hough = np.zeros((rho_max, 180), dtype=int)

        # get index of edge
        # ind[0] 是 符合条件的纵坐标，ind[1]是符合条件的横坐标
        ind = np.where(edge == 255)

        ## hough transformation
        # zip函数返回元组
        for y, x in zip(ind[0], ind[1]):
            for theta in range(0, 180, dtheta):
                # get polar coordinat4s
                t = np.pi / 180 * theta
                rho = int(x * np.cos(t) + y * np.sin(t))

                # vote
                hough[rho, theta] += 1

        out = hough.astype(np.uint8)

        return out

    # non maximum suppression
    def non_maximum_suppression(hough):
        rho_max, _ = hough.shape

        ## non maximum suppression
        for y in range(rho_max):
            for x in range(180):
                # get 8 nearest neighbor
                x1 = max(x - 1, 0)
                x2 = min(x + 2, 180)
                y1 = max(y - 1, 0)
                y2 = min(y + 2, rho_max - 1)
                if np.max(hough[y1:y2, x1:x2]) == hough[y, x] and hough[y, x] != 0:
                    pass
                    # hough[y,x] = 255
                else:
                    hough[y, x] = 0

        return hough

    def inverse_hough(hough, img):
        H, W, _ = img.shape
        rho_max, _ = hough.shape

        out = img.copy()

        # get x, y index of hough table
        # np.ravel 将多维数组降为1维
        # argsort  将数组元素从小到大排序，返回索引
        # [::-1]   反序->从大到小
        # [:20]    前20个
        ind_x = np.argsort(hough.ravel())[::-1][:80]
        ind_y = ind_x.copy()
        thetas = ind_x % 180
        rhos = ind_y // 180
        lines = []
        for theta, rho in zip(thetas, rhos):
            t = np.pi / 180. * theta
            if - (np.cos(t) / np.sin(t)) > 0:
                x1, x2 = 300, 400
            elif - (np.cos(t) / np.sin(t)) < 0:
                x1, x2 = 110, 120
            if np.sin(t) != 0:
                y1 = - (np.cos(t) / np.sin(t)) * x1 + rho / np.sin(t)
                y2 = - (np.cos(t) / np.sin(t)) * x2 + rho / np.sin(t)
                y1 = int(y1)
                y2 = int(y2)
                line = [[x1, y1], [x2, y2]]
                lines.append(line)
        return lines

    # voting
    hough = voting(edge)

    # non maximum suppression
    hough = non_maximum_suppression(hough)

    # inverse hough
    out = inverse_hough(hough, img)

    return out
