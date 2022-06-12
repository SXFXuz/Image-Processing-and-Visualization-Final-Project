import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import cv2
from plot_utils import mask_img_from_phi, select_region

# 曲线演化
def cv_evolution(phi, u, mu, v, lambda1, lambda2, delta_t, iter_num):
    """
        Chan-Vese模型 曲线演化 迭代iter_num次
        :param phi: 水平集函数phi
        :param u: 图像灰度值矩阵
        :param mu: 公式 E 中，曲线长度项前的系数
        :param v: 公式 E 中，曲线围成面积项前的系数
        :param lambda1: 公式 E 中，第一个方差项前的系数
        :param lambda2: 公式 E 中，第二个方差项前的系数
        :param delta_t: 步长
        :param iter_num: 迭代次数
        :return: 演化后的phi
        """
    eps = 1
    for i in range(iter_num):
        phi = reinitialization(phi, 5)
        phi_y, phi_x = np.gradient(phi)
        grad_norm = np.sqrt(phi_x**2 + phi_y**2)

        phi_x = phi_x/(grad_norm + 1e-6)
        phi_y = phi_y/(grad_norm + 1e-6)

        c1 = np.mean(u[phi>0])
        c2 = np.mean(u[phi<0])

        Mxx, Nxx = np.gradient(phi_x)
        Nyy, Myy = np.gradient(phi_y)
        divergence = Nxx + Nyy
        phi_t = mu * divergence - v - lambda1 * (u - c1) **2 + lambda2 * (u - c2) ** 2
        delta_phi = eps/(np.pi * (eps**2 + phi**2))

        phi = phi + delta_t * phi_t * delta_phi

    return phi


# 重新初始化
def reinitialization(phi, iter):
    S = phi/np.sqrt(phi**2 + 1)
    for i in range(iter):
        phi_x_p = Dx_p(phi)
        phi_x_m = Dx_m(phi)
        phi_y_p = Dy_p(phi)
        phi_y_m = Dy_m(phi)

        T = S.copy()
        T[T > 0] = 1
        T[T < 0] = -1

        T_max_0 = np.where(T>0, T, 0)
        T_min_0 = np.where(T<0, T, 0)
        phiX_maxM_sq = np.where(phi_x_m>0, phi_x_m, 0)**2
        phiX_minM_sq = np.where(phi_x_m < 0, phi_x_m, 0)**2
        phiX_maxP_sq = np.where(phi_x_p > 0, phi_x_p, 0)**2
        phiX_minP_sq = np.where(phi_x_p<0, phi_x_p, 0)**2
        temp1 = np.where(phiX_maxM_sq > phiX_minP_sq, phiX_maxM_sq, phiX_minP_sq)
        temp2 = np.where(phiX_minM_sq > phiX_maxP_sq, phiX_minM_sq, phiX_maxP_sq)

        phiY_maxM_sq = np.where(phi_y_m > 0, phi_y_m, 0) ** 2
        phiY_minM_sq = np.where(phi_y_m < 0, phi_y_m, 0) ** 2
        phiY_maxP_sq = np.where(phi_y_p > 0, phi_y_p, 0) ** 2
        phiY_minP_sq = np.where(phi_y_p < 0, phi_y_p, 0) ** 2
        temp3 = np.where(phiY_maxM_sq > phiY_minP_sq, phiY_maxM_sq, phiY_minP_sq)
        temp4 = np.where(phiY_minM_sq > phiY_maxP_sq, phiY_minM_sq, phiY_maxP_sq)

        phi_x_sq = T_max_0 * temp1 - T_min_0 * temp2
        phi_y_sq = T_max_0 * temp3 - T_min_0 * temp4

        abs_grad_phi = np.sqrt(phi_x_sq + phi_y_sq)
        abs_H1 = np.abs(S * np.sqrt(phi_x_sq) / (abs_grad_phi + (abs_grad_phi == 0)))
        abs_H2 = np.abs(S * np.sqrt(phi_y_sq) / (abs_grad_phi + (abs_grad_phi == 0)))
        max_H1_H2 = np.max(abs_H1 + abs_H2)

        delta_t = 1 / (max_H1_H2 + (max_H1_H2 == 0))
        phi = phi + delta_t * (-S * abs_grad_phi + S)

    return phi


# 使用差分定义各方向上的derivatives
def Dx_p(phi):
    B = np.roll(phi, -1, axis=1)
    B[:, -1] = phi[:, -1]
    return B - phi

def Dx_m(phi):
    B = np.roll(phi, 1, axis=1)
    B[:, 0] = phi[:, 0]
    return phi - B


def Dy_p(phi):
    C = np.roll(phi, -1, axis=0)
    C[-1, :] = phi[-1, :]
    return C - phi


def Dy_m(phi):
    C = np.roll(phi, 1, axis=0)
    C[0, :] = phi[0, :]
    return phi - C


if __name__ == '__main__':
    # 图片
    img = cv2.imread("./testcase/cv_image.bmp")

    # 选择初始水平集
    phi = select_region(img)

    img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    u = img_grey
    u = (u - np.min(u)) / (np.max(u) - np.min(u)) * 255

    # 参数
    iteration = 600
    iter = 0
    mu = 0.001 * (255 ** 2)
    lambda1 = 1  # outside uniformity
    lambda2 = 1  # inside uniformity
    delta_t = 0.0001
    v = 0

    # 初始边界展示
    # plt.figure(1)
    # plt.imshow(mask_img_from_phi(img, phi))
    plt.figure(1), plt.imshow(img), plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.contour(phi, [0], color='b')  # 画LSF=0处的等高线
    plt.draw(), plt.show(block=False), plt.pause(3)
    plt.close()

    # 迭代
    while iter < iteration:
        phi = cv_evolution(phi, u, mu, v, lambda1, lambda2, delta_t, 5)
        iter += 5

        # plt.figure(1)
        # plt.imshow(mask_img_from_phi(img, phi))
        # plt.pause(0.01)
        plt.figure(3), plt.imshow(img), plt.xticks([]), plt.yticks([])
        plt.title('Iteration: {}'.format(iter))
        h = plt.contour(phi, [0], colors='r')
        plt.draw(), plt.show(block=False), plt.pause(0.2)
        plt.clf()

        print(iter)

