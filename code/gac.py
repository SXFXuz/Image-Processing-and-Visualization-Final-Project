import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from plot_utils import select_region, mask_img_from_phi
import cv2


# copy之前作业的函数(以下三个），用来算个卷积
def conv2d(X, K):
    """
    实现二维卷积运算
    :param X: 二维数组
    :param K: 卷积核，二维数组
    :return: 卷积运算的结果，二维数组
    """
    h, w = K.shape
    K = np.flipud(K)
    K = np.fliplr(K)
    Y = np.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for row in range(Y.shape[0]):
        for col in range(Y.shape[1]):
            Y[row, col] = np.sum(X[row : row + h, col : col + w] * K)
    return Y


def padding(img, kernel):
    """
    填充使得卷积操作时核的每个元素能够访问图像的每个像素点，便于实现卷积的可交换性等性质
    :param img: 图像灰度值矩阵
    :param kernel: 卷积核
    :return: 填充后的图像
    """
    n_h, n_w = img.shape
    k_h, k_w = kernel.shape
    p_h = 2 * (k_h - 1)
    p_w = 2 * (k_w - 1)
    img2 = np.zeros(shape=(n_h + p_h, n_w + p_w))
    ph_start = k_h - 1
    pw_start = k_w - 1
    img2[ph_start : ph_start + n_h, pw_start : pw_start + n_w] = img
    return img2


def guassion_kernel(h, w, sigma=1.0, K=1.0):
    """
    高斯卷积核
    :param h: 卷积核的长度
    :param w: 卷积核的宽度
    :param sigma: 公式中的sigma
    :param K: 公式中的指数前的系数K
    :return: 卷积核，二维数组
    """
    center = h // 2, w // 2
    kernel = np.zeros(shape=(h, w), dtype=float)
    for row in range(h):
        for col in range(w):
            kernel[row, col] = K * np.exp(
                -((row - center[0]) ** 2 + (col - center[1]) ** 2) / (2 * (sigma ** 2))
            )
    kernel = kernel / np.sum(kernel)
    return kernel


# copy matlab里那个算图像卷积的函数
def create_g(img):
    G = guassion_kernel(3, 3)
    Gx, Gy = np.gradient(G)
    img2 = padding(img, G)
    g = conv2d(img2, Gx) ** 2 + conv2d(img2, Gy) ** 2
    g = g[1:-1, 1:-1]
    return 1 / (1 + g)


# 单步更新函数
def gac_evolution(phi, g, gx, gy, delta_t):
    c = 1
    phi = reinitialization(phi, 20)
    phi = lsm_evolution(phi, -g * c, -gx, -gy, g, 1, delta_t)
    return phi


# 重新初始化
def reinitialization(phi, iter):
    S = phi / np.sqrt(phi ** 2 + 1)
    for i in range(iter):
        phi_x_p = Dx_p(phi)
        phi_x_m = Dx_m(phi)
        phi_y_p = Dy_p(phi)
        phi_y_m = Dy_m(phi)

        T = S.copy()
        T[T > 0] = 1
        T[T < 0] = -1

        T_max_0 = np.where(T > 0, T, 0)
        T_min_0 = np.where(T < 0, T, 0)
        phiX_maxM_sq = np.where(phi_x_m > 0, phi_x_m, 0) ** 2
        phiX_minM_sq = np.where(phi_x_m < 0, phi_x_m, 0) ** 2
        phiX_maxP_sq = np.where(phi_x_p > 0, phi_x_p, 0) ** 2
        phiX_minP_sq = np.where(phi_x_p < 0, phi_x_p, 0) ** 2
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


# general hamilton-jacobi equation for level set method
def lsm_evolution(phi, a, u, v, b, iter, alpha=1):
    """
        水平集法对 level set equation 进行迭代求解
        :param phi: 水平集函数 phi
        :param a: -图像灰度值矩阵g（经过高斯核模糊）
        :param u: g对 x 的偏导数
        :param v: g对 y 的偏导数
        :param b: 图像灰度值矩阵
        :param iter: 迭代次数
        :param alpha: 用于调整 CLF 条件
        :return: 迭代后的水平集函数 phi
        """
    for i in range(iter):
        phi_x_p = Dx_p(phi)  # Dx +
        phi_x_m = Dx_m(phi)  # Dx -
        phi_x_o = Dx_o(phi)  # Dx  [f(x+1)-f(x-1)]/2
        phi_y_p = Dy_p(phi)  # Dy +
        phi_y_m = Dy_m(phi)  # Dy -
        phi_y_o = Dy_o(phi)  # Dy

        T = a.copy()
        T[T > 0] = 1
        T[T < 0] = -1

        T_max_0 = np.where(T > 0, T, 0)
        T_min_0 = np.where(T < 0, T, 0)
        phiX_maxM_sq = np.where(phi_x_m > 0, phi_x_m, 0) ** 2  # max(Dx-,0)**2
        phiX_minM_sq = np.where(phi_x_m < 0, phi_x_m, 0) ** 2  # min(Dx-,0)**2
        phiX_maxP_sq = np.where(phi_x_p > 0, phi_x_p, 0) ** 2  # max(Dx+,0)**2
        phiX_minP_sq = np.where(phi_x_p < 0, phi_x_p, 0) ** 2  # min(Dx+,0)**2

        phiY_maxM_sq = np.where(phi_y_m > 0, phi_y_m, 0) ** 2  # 同上,对y
        phiY_minM_sq = np.where(phi_y_m < 0, phi_y_m, 0) ** 2
        phiY_maxP_sq = np.where(phi_y_p > 0, phi_y_p, 0) ** 2
        phiY_minP_sq = np.where(phi_y_p < 0, phi_y_p, 0) ** 2

        phi_x_sq = T_max_0 * (phiX_maxM_sq + phiX_minP_sq) - T_min_0 * (
            phiX_minM_sq + phiX_maxP_sq
        )  # 计算 phi 的梯度
        phi_y_sq = T_max_0 * (phiY_maxM_sq + phiY_minP_sq) - T_min_0 * (
            phiY_minM_sq + phiY_maxP_sq
        )
        abs_grad_phi_upwind = np.sqrt(phi_x_sq + phi_y_sq)

        u_max0 = np.where(u > 0, u, 0)
        u_min0 = np.where(u < 0, u, 0)
        v_max0 = np.where(v > 0, v, 0)
        v_min0 = np.where(v < 0, v, 0)
        # upwind scheme
        convection_upwind = (
            u_max0 * phi_x_m + u_min0 * phi_x_p + v_max0 * phi_y_m + v_min0 * phi_y_p
        )
        abs_grad_phi_central = np.sqrt(phi_x_o ** 2 + phi_y_o ** 2)

        # 计算曲率
        kappa = curvature(phi)
        # CFL 条件
        abs_H1 = np.abs(
            u
            + a * np.sqrt(phi_x_sq) / (abs_grad_phi_upwind + (abs_grad_phi_upwind == 0))
        )
        abs_H2 = np.abs(
            v
            + a * np.sqrt(phi_y_sq) / (abs_grad_phi_upwind + (abs_grad_phi_upwind == 0))
        )
        max_H1_H2_2b_2b = np.max(abs_H1 + abs_H2 + 2 * b + 2 * b)
        delta_t = alpha / (max_H1_H2_2b_2b + (max_H1_H2_2b_2b == 0))
        # 迭代过程
        phi = phi + delta_t * (
            -a * abs_grad_phi_upwind
            - convection_upwind
            + b * kappa * abs_grad_phi_central
        )
    return phi


# 计算曲率
def curvature(phi):
    phi_x = Dx_o(phi)
    phi_y = Dy_o(phi)
    phi_xx = Dxx(phi)
    phi_yy = Dyy(phi)
    phi_xy = Dxy(phi)

    abs_grad_phi_sq = phi_x * phi_x + phi_y * phi_y
    abs_grad_phi_cube = abs_grad_phi_sq ** 1.5

    kappa = (
        phi_xx * phi_y * phi_y - 2 * phi_y * phi_x * phi_xy + phi_yy * phi_x * phi_x
    ) / (abs_grad_phi_cube + (abs_grad_phi_cube < 1e-4))
    return kappa


# shift将矩阵沿某个方向平移一个单位
def shift_L(phi):
    A = np.roll(phi, -1, axis=1)
    A[:, -1] = phi[:, -1]
    return A


def shift_R(phi):
    B = np.roll(phi, 1, axis=1)
    B[:, 0] = phi[:, 0]
    return B


def shift_U(phi):
    C = np.roll(phi, -1, axis=0)
    C[-1, :] = phi[-1, :]
    return C


def shift_D(phi):
    D = np.roll(phi, 1, axis=0)
    D[0, :] = phi[0, :]
    return D


# 计算梯度, 有这几种: 1) x(n) - x(n-1)  2) x(n+1) - x(n)  3) [x(n+1) - x(n-1)]/2
def Dx_p(phi):
    A = shift_L(phi)
    return A - phi


def Dx_m(phi):
    B = shift_R(phi)
    return phi - B


def Dx_o(phi):
    A = shift_L(phi)
    B = shift_R(phi)
    return (A - B) / 2


def Dy_p(phi):
    C = shift_U(phi)
    return C - phi


def Dy_m(phi):
    D = shift_D(phi)
    return phi - D


def Dy_o(phi):
    C = shift_U(phi)
    D = shift_D(phi)
    return (C - D) / 2


#  计算二阶偏导数
def Dxx(phi):
    A = shift_L(phi)
    B = shift_R(phi)
    return A - 2 * phi + B


def Dyy(phi):
    C = shift_U(phi)
    D = shift_D(phi)
    return C - 2 * phi + D


def Dxy(phi):
    D_xy = (
        shift_R(shift_D(phi))
        - shift_L(shift_D(phi))
        + shift_L(shift_U(phi))
        - shift_R(shift_U(phi))
    ) / 4
    return D_xy


if __name__ == '__main__':
    # 图片
    img = cv2.imread("./testcase/cv_image.bmp")

    # 选择初始水平集
    phi = select_region(img)
    img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    u = img_grey
    u = (u - np.min(u)) / (np.max(u) - np.min(u)) * 255

    # 初始边界展示
    # plt.figure(1)
    # plt.imshow(mask_img_from_phi(img, phi))
    plt.figure(1), plt.imshow(img), plt.xticks([]), plt.yticks(
        []
    )  # to hide tick values on X and Y axis
    plt.contour(phi, [0], color='b')  # 画LSF=0处的等高线
    plt.draw(), plt.show(block=False), plt.pause(3)
    plt.close()

    # 先对图像平滑
    g = create_g(u)
    gy, gx = np.gradient(g)

    # 参数
    max_iter = 300
    it = 0
    delta_t = 5

    # 循环迭代
    while it <= max_iter:
        phi = gac_evolution(phi, g, gx, gy, delta_t)
        it += 1

        # plt.figure(1)
        # plt.imshow(mask_img_from_phi(img, phi))

        plt.figure(3), plt.imshow(img), plt.xticks([]), plt.yticks([])
        plt.title('Iteration: {}'.format(it))
        h = plt.contour(phi, [0], colors='r')
        plt.draw(), plt.show(block=False), plt.pause(0.2)
        plt.clf()

        # plt.pause(0.01)
