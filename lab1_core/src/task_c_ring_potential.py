import math
import numpy as np

def axis_potential_analytic(z: float, a: float = 1.0, q: float = 1.0) -> float:
    """
    均匀带电圆环轴线上电势的解析解
    V(z) = q / (2 * a) * 1 / sqrt(1 + (z/a)^2)
    （a=1, q=1时，简化为V(z) = 1 / (2 * sqrt(1 + z^2))）
    """
    return q / (2 * math.sqrt(a**2 + z**2))

def ring_potential_point(x: float, y: float, z: float, a: float = 1.0, q: float = 1.0, n_phi: int = 720) -> float:
    """
    计算均匀带电圆环在空间点(x,y,z)的电势
    积分公式: V = q/(2π) ∫0^2π dφ / sqrt( (x - a cosφ)^2 + (y - a sinφ)^2 + z^2 )
    :param x,y,z: 空间点坐标
    :param a: 圆环半径，默认1.0
    :param q: 总电荷，默认1.0
    :param n_phi: φ方向分段数，默认720
    :return: 电势值
    """
    dphi = 2 * math.pi / n_phi
    V = 0.0
    for i in range(n_phi):
        phi = i * dphi
        dx = x - a * math.cos(phi)
        dy = y - a * math.sin(phi)
        r = math.sqrt(dx**2 + dy**2 + z**2)
        V += 1.0 / r
    V = V * dphi * q / (2 * math.pi)
    return V

def ring_potential_grid(ys: np.ndarray, zs: np.ndarray, x0: float = 0.0, a: float = 1.0, q: float = 1.0, n_phi: int = 360) -> np.ndarray:
    """
    计算yz平面（x=x0）的电势网格
    :param ys: y方向坐标数组
    :param zs: z方向坐标数组
    :param x0: x坐标，默认0.0（yz平面）
    :param a: 圆环半径，默认1.0
    :param q: 总电荷，默认1.0
    :param n_phi: φ方向分段数，默认360
    :return: 电势网格V，形状为(len(zs), len(ys))
    """
    ny = len(ys)
    nz = len(zs)
    V = np.zeros((nz, ny))
    for i in range(nz):
        z = zs[i]
        for j in range(ny):
            y = ys[j]
            V[i, j] = ring_potential_point(x0, y, z, a, q, n_phi)
    return V
