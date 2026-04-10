import numpy as np

G = 6.674e-11

def gauss_legendre_2d(func, ax: float, bx: float, ay: float, by: float, n: int = 40) -> float:
    # 一维高斯-勒让德节点和权重
    nodes, weights = np.polynomial.legendre.leggauss(n)
    # 映射到实际积分区间
    x_map = 0.5 * (bx - ax) * nodes + 0.5 * (ax + bx)
    y_map = 0.5 * (by - ay) * nodes + 0.5 * (ay + by)
    # 雅可比行列式
    jacobian = 0.25 * (bx - ax) * (by - ay)
    # 张量积计算二重积分
    X, Y = np.meshgrid(x_map, y_map)
    Wx, Wy = np.meshgrid(weights, weights)
    f_vals = func(X, Y)
    integral = np.sum(f_vals * Wx * Wy) * jacobian
    return integral

def plate_force_z(z: float, L: float = 10.0, M_plate: float = 1.0e4, m_particle: float = 1.0, n: int = 40) -> float:
    if z < 0:
        raise ValueError("z must be non-negative (above the plate)")
    # 面密度
    sigma = M_plate / (L ** 2)
    # 积分区间：[-L/2, L/2] × [-L/2, L/2]
    ax, bx = -L/2, L/2
    ay, by = -L/2, L/2
    
    # 被积函数
    def integrand(x, y):
        r_sq = x**2 + y**2 + z**2
        return z / (r_sq ** (3/2))
    
    # 二维高斯积分
    integral = gauss_legendre_2d(integrand, ax, bx, ay, by, n)
    # 计算引力
    Fz = G * sigma * m_particle * integral
    return Fz

def force_curve(z_values, L: float = 10.0, M_plate: float = 1.0e4, m_particle: float = 1.0, n: int = 40):
    Fz_list = []
    for z in z_values:
        Fz = plate_force_z(z, L, M_plate, m_particle, n)
        Fz_list.append(Fz)
    return np.array(Fz_list)
