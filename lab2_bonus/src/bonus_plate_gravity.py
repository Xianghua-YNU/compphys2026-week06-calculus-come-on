import numpy as np
import matplotlib.pyplot as plt

# 万有引力常数 (m³ kg⁻¹ s⁻²)
G = 6.674e-11


def gauss_legendre_1d(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    生成1维n点高斯-勒让德积分的节点和权重
    支持n=2,3,4,5,6,7,8,9,10(常用高精度节点)
    """
    # 预定义常用n点的节点和权重（来自高斯-勒让德积分表）
    gl_table = {
        2: (np.array([-0.5773502691896257, 0.5773502691896257]),
            np.array([1.0, 1.0])),
        3: (np.array([-0.7745966692414834, 0.0, 0.7745966692414834]),
            np.array([0.5555555555555556, 0.8888888888888888, 0.5555555555555556])),
        4: (np.array([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526]),
            np.array([0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538])),
        5: (np.array([-0.9061798459386640, -0.5384693101056830, 0.0, 0.5384693101056830, 0.9061798459386640]),
            np.array([0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891])),
        8: (np.array([-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834641831709973,
                      0.1834641831709973, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363]),
            np.array([0.1012285362903763, 0.2223810344533743, 0.3137066458778873, 0.3626837833783620,
                      0.3626837833783620, 0.3137066458778873, 0.2223810344533743, 0.1012285362903763]))
    }
    if n not in gl_table:
        raise ValueError(f"Unsupported n={n}, supported n: {list(gl_table.keys())}")
    return gl_table[n]


def gauss_legendre_2d(func, ax: float, bx: float, ay: float, by: float, n: int = 8) -> float:
    """
    二维高斯-勒让德积分：计算∫(ax到bx)∫(ay到by) func(x,y) dxdy
    采用张量积形式：将区间[-1,1]*[-1,1]映射到[ax,bx]*[ay,by]
    :param func: 被积函数 f(x,y)
    :param ax,bx: x积分上下限
    :param ay,by: y积分上下限
    :param n: 高斯点数量(默认8点,高精度)
    :return: 积分结果
    """
    # 获取1维高斯节点和权重
    x_nodes, x_weights = gauss_legendre_1d(n)
    y_nodes, y_weights = gauss_legendre_1d(n)

    # 区间映射：[-1,1] → [a,b]，变换公式：t = (b-a)/2 * s + (a+b)/2
    hx = (bx - ax) / 2.0
    cx = (ax + bx) / 2.0
    hy = (by - ay) / 2.0
    cy = (ay + by) / 2.0

    # 张量积积分
    integral = 0.0
    for i in range(n):
        x = hx * x_nodes[i] + cx
        wx = x_weights[i]
        for j in range(n):
            y = hy * y_nodes[j] + cy
            wy = y_weights[j]
            integral += wx * wy * func(x, y)

    # 乘以雅可比行列式（区间变换的缩放因子）
    integral *= hx * hy
    return integral


def plate_force_z(z: float, L: float = 10.0, M_plate: float = 1.0e4, m_particle: float = 1.0, n: int = 8) -> float:
    """
    计算正方形金属板中心正上方z处,质点受到的z方向万有引力
    公式:Fz(z) = G * σ * m_particle * z ∫∫ dxdy / (x² + y² + z²)^(3/2)
    其中σ = M_plate / L² 是面密度,积分区间x∈[-L/2, L/2], y∈[-L/2, L/2]
    :param z: 质点到板的垂直距离 (m),z>0
    :param L: 方板边长 (m),默认10.0
    :param M_plate: 方板总质量 (kg),默认1e4(10吨)
    :param m_particle: 质点质量 (kg),默认1.0
    :param n: 高斯积分点数,默认8
    :return: Fz (N),方向沿z轴正方向(指向板)
    """
    if z <= 0:
        raise ValueError("z must be positive (above the plate)")

    # 面密度 σ = M / L²
    sigma = M_plate / (L ** 2)

    # 定义被积函数 f(x,y) = z / (x² + y² + z²)^(3/2)
    def integrand(x, y):
        r_sq = x**2 + y**2 + z**2
        return z / (r_sq ** (3/2))

    # 积分区间：x∈[-L/2, L/2], y∈[-L/2, L/2]
    ax, bx = -L/2, L/2
    ay, by = -L/2, L/2

    # 二维高斯积分
    integral = gauss_legendre_2d(integrand, ax, bx, ay, by, n=n)

    # 计算Fz = G * σ * m_particle * integral
    Fz = G * sigma * m_particle * integral
    return Fz


def force_curve(z_values: np.ndarray, L: float = 10.0, M_plate: float = 1.0e4,
                m_particle: float = 1.0, n: int = 8) -> np.ndarray:
    """
    计算z_values数组对应的Fz数组
    """
    Fz = np.zeros_like(z_values)
    for i, z in enumerate(z_values):
        Fz[i] = plate_force_z(z, L, M_plate, m_particle, n)
    return Fz


def plot_force_curve(z_values: np.ndarray, Fz: np.ndarray):
    """
    绘制Fz随z变化的曲线
    """
    plt.figure(figsize=(10, 6))
    plt.plot(z_values, Fz, 'b-', linewidth=2, label='Gravitational Force Fz')
    plt.scatter(z_values, Fz, c='red', s=50, zorder=5)

    # 标注关键点
    key_z = [0.2, 1.0, 5.0, 10.0]
    for z in key_z:
        idx = np.argmin(np.abs(z_values - z))
        f = Fz[idx]
        plt.annotate(f'z={z}m\nF={f:.2e}N', xy=(z, f), xytext=(z+0.5, f*1.1),
                     arrowprops=dict(arrowstyle='->', color='black'), fontsize=10)

    plt.xlabel('Distance z (m)', fontsize=12)
    plt.ylabel('Gravitational Force Fz (N)', fontsize=12)
    plt.title('Gravitational Force on 1kg Mass Above a 10m Square Plate', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('plate_force_curve.png', dpi=300, bbox_inches='tight')
    plt.show()


# ------------------- 测试与验证 -------------------
if __name__ == "__main__":
    # 1. 计算要求的z点
    z_test = np.array([0.2, 1.0, 5.0, 10.0])
    Fz_test = force_curve(z_test)
    print("z (m) | Fz (N)")
    print("----------------")
    for z, f in zip(z_test, Fz_test):
        print(f"{z:5.1f}  | {f:.4e}")

    # 2. 绘制完整曲线（z∈[0.2,10]）
    z_full = np.linspace(0.2, 10.0, 100)
    Fz_full = force_curve(z_full)
    plot_force_curve(z_full, Fz_full)

    # 3. 验证远场近似：z>>L时，方板近似为质点，Fz≈G*M*m/z²
    z_large = 100.0
    Fz_num = plate_force_z(z_large)
    Fz_approx = G * M_plate * m_particle / (z_large ** 2)
    print(f"\n远场验证 (z=100m):")
    print(f"数值解: {Fz_num:.4e} N, 质点近似: {Fz_approx:.4e} N, 相对误差: {np.abs(Fz_num-Fz_approx)/Fz_approx:.2e}")
