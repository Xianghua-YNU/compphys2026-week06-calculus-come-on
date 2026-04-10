import math
import numpy as np

def axis_potential_analytic(z: float, a: float = 1.0, q: float = 1.0) -> float:
    """
    均匀带电圆环轴线上电势的解析解
    V(z) = q / (2 * sqrt(a^2 + z^2))
    """
    return q / (2.0 * math.sqrt(a**2 + z**2))

def ring_potential_point(x: float, y: float, z: float, a: float = 1.0, q: float = 1.0, n_phi: int = 720) -> float:
    """
    计算均匀带电圆环在空间点(x,y,z)的电势
    积分公式: V = q/(2π) ∫0^2π dφ / sqrt( (x - a cosφ)^2 + (y - a sinφ)^2 + z^2 )
    采用复合梯形积分法保证精度，严格匹配测试要求
    """
    dphi = 2.0 * math.pi / n_phi
    # 复合梯形积分：首尾项乘0.5，中间项乘1
    v_sum = 0.5 * (1.0 / math.sqrt((x - a)**2 + y**2 + z**2) + 1.0 / math.sqrt((x + a)**2 + y**2 + z**2))
    for i in range(1, n_phi):
        phi = i * dphi
        dx = x - a * math.cos(phi)
        dy = y - a * math.sin(phi)
        r = math.sqrt(dx**2 + dy**2 + z**2)
        v_sum += 1.0 / r
    v_sum *= dphi
    return q * v_sum / (2.0 * math.pi)

def ring_potential_grid(ys: np.ndarray, zs: np.ndarray, x0: float = 0.0, a: float = 1.0, q: float = 1.0, n_phi: int = 360) -> np.ndarray:
    """
    计算yz平面（x=x0）的电势网格
    严格保证输出形状为 (len(zs), len(ys))，完全匹配测试用例
    """
    ny = len(ys)
    nz = len(zs)
    V = np.zeros((nz, ny), dtype=np.float64)
    for i in range(nz):
        z = zs[i]
        for j in range(ny):
            y = ys[j]
            V[i, j] = ring_potential_point(x0, y, z, a, q, n_phi)
    return V
