import numpy as np
import matplotlib.pyplot as plt

# 万有引力常数
G = 6.674e-11

# 1. 一维高斯-勒让德积分节点和权重（预计算）
def gauss_legendre_1d(n):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return nodes, weights

# 2. 二维高斯-勒让德积分（矩形区域 [ax, bx] × [ay, by]）
def gauss_legendre_2d(func, ax: float, bx: float, ay: float, by: float, n: int = 40) -> float:
    # 获取一维节点和权重
    x_nodes, x_weights = gauss_legendre_1d(n)
    y_nodes, y_weights = gauss_legendre_1d(n)
    
    # 映射到实际积分区间
    x_map = 0.5 * (bx - ax) * x_nodes + 0.5 * (ax + bx)
    y_map = 0.5 * (by - ay) * y_nodes + 0.5 * (ay + by)
    
    # 雅可比行列式
    jacobian = 0.25 * (bx - ax) * (by - ay)
    
    # 张量积计算二重积分（向量化，避免循环）
    X, Y = np.meshgrid(x_map, y_map)
    Wx, Wy = np.meshgrid(x_weights, y_weights)
    f_vals = func(X, Y)
    integral = np.sum(f_vals * Wx * Wy) * jacobian
    return integral

# 3. 方板中心正上方z处的引力z分量
def plate_force_z(z: float, L: float = 10.0, M_plate: float = 1.0e4, m_particle: float = 1.0, n: int = 40) -> float:
    if z < 0:
        raise ValueError("z must be non-negative (above the plate)")
    # 面密度
    sigma = M_plate / (L ** 2)
    # 积分区间：[-L/2, L/2] × [-L/2, L/2]
    ax, bx = -L/2, L/2
    ay, by = -L/2, L/2
    
    # 定义被积函数
    def integrand(x, y):
        r_sq = x**2 + y**2 + z**2
        return z / (r_sq ** (3/2))
    
    # 二维高斯-勒让德积分
    integral = gauss_legendre_2d(integrand, ax, bx, ay, by, n)
    # 计算最终引力
    Fz = G * sigma * m_particle * integral
    return Fz

# 4. 生成z对应的Fz数组
def force_curve(z_values, L: float = 10.0, M_plate: float = 1.0e4, m_particle: float = 1.0, n: int = 40):
    Fz_list = []
    for z in z_values:
        Fz = plate_force_z(z, L, M_plate, m_particle, n)
        Fz_list.append(Fz)
    return np.array(Fz_list)

# 5. 可视化引力随z的变化
def plot_force_curve():
    z_values = np.linspace(0.2, 10.0, 50)
    Fz = force_curve(z_values)
    
    plt.figure(figsize=(8, 5))
    plt.plot(z_values, Fz, "b-", linewidth=2, marker="o", markersize=4)
    plt.xlabel("z (m)")
    plt.ylabel("Gravitational Force Fz (N)")
    plt.title("Gravitational Force on 1kg Mass Above 10m Square Plate (10 Tonnes)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plate_gravity_force.png", dpi=300, bbox_inches="tight")
    plt.show()

# 自测运行
if __name__ == "__main__":
    # 测试z=1m处的引力
    z_test = 1.0
    Fz_test = plate_force_z(z_test)
    print(f"Bonus z={z_test}m处的引力Fz: {Fz_test:.6e} N")
    # 绘制z∈[0.2,10]的力曲线
    plot_force_curve()
