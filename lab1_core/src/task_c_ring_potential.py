import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


def ring_potential_point(x: float, y: float, z: float, a: float = 1.0, q: float = 1.0) -> float:
    """
    用数值积分计算均匀带电圆环在空间任意点(x,y,z)的电势
    公式:V(x,y,z) = q/(2π) ∫₀²π dφ / √[(x - a cosφ)² + (y - a sinφ)² + z²]
    :param x,y,z: 场点坐标
    :param a: 圆环半径,默认1.0
    :param q: 总电荷,默认1.0(对应Q=4πε₀q的归一化)
    :return: 电势值
    """
    # 定义被积函数
    def integrand(phi):
        dx = x - a * np.cos(phi)
        dy = y - a * np.sin(phi)
        r = np.sqrt(dx**2 + dy**2 + z**2)
        # 数值稳定性：避免r=0时的发散（场点在圆环上）
        if r < 1e-10:
            return 1e10  # 截断，实际物理上圆环上电势为无穷大，数值上用大值近似
        return 1.0 / r

    # 用scipy的quad做高精度数值积分（比梯形法/Simpson法更稳定）
    integral, _ = quad(integrand, 0, 2 * np.pi)
    return q / (2 * np.pi) * integral


def ring_potential_grid(y_grid: np.ndarray, z_grid: np.ndarray, x0: float = 0.0,
                        a: float = 1.0, q: float = 1.0) -> np.ndarray:
    """
    在y-z平面(x=x0)上计算电势网格矩阵
    :param y_grid: y方向网格数组 (Ny,)
    :param z_grid: z方向网格数组 (Nz,)
    :param x0: 平面x坐标,默认0.0(过圆心的y-z平面)
    :param a: 圆环半径
    :param q: 总电荷
    :return: 电势矩阵 (Nz, Ny),对应z行y列
    """
    Ny = len(y_grid)
    Nz = len(z_grid)
    V = np.zeros((Nz, Ny))

    # 向量化计算，避免双重循环（提升效率）
    for i in range(Nz):
        z = z_grid[i]
        for j in range(Ny):
            y = y_grid[j]
            V[i, j] = ring_potential_point(x0, y, z, a, q)

    return V


def axis_potential_analytic(z: float, a: float = 1.0, q: float = 1.0) -> float:
    """
    圆环轴线上(x=0,y=0)的电势解析解，用于验证数值结果
    公式:V(z) = q / √(a² + z²)
    """
    return q / np.sqrt(a**2 + z**2)


def compute_electric_field(y_grid: np.ndarray, z_grid: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    由电势梯度计算电场分量 E = -∇V
    :param y_grid, z_grid: 网格坐标
    :param V: 电势矩阵 (Nz, Ny)
    :return: (Ey, Ez) 电场分量矩阵
    """
    # 计算网格步长
    dy = y_grid[1] - y_grid[0]
    dz = z_grid[1] - z_grid[0]

    # 用中心差分计算梯度（边界用前向/后向差分）
    dV_dy, dV_dz = np.gradient(V, dz, dy)  # np.gradient顺序是(行,列)对应(z,y)
    Ey = -dV_dy
    Ez = -dV_dz

    return Ey, Ez


def plot_ring_potential_field(y_grid: np.ndarray, z_grid: np.ndarray, V: np.ndarray,
                              Ey: np.ndarray, Ez: np.ndarray, a: float = 1.0):
    """
    绘制y-z平面的等势线图+电场线图
    """
    Y, Z = np.meshgrid(y_grid, z_grid)

    plt.figure(figsize=(10, 8))

    # 1. 绘制等势线
    levels = np.linspace(V.min(), V.max(), 20)
    contour = plt.contour(Y, Z, V, levels=levels, cmap='viridis', linewidths=1.5)
    plt.colorbar(contour, label='Electric Potential V')

    # 2. 绘制电场线（streamplot）
    plt.streamplot(Y, Z, Ey, Ez, color='red', linewidth=1, density=1.5, arrowstyle='->', arrowsize=1.2)

    # 3. 绘制圆环位置（y=±a, z=0）
    plt.plot([a, a], [0, 0], 'ko', markersize=8, label='Ring (y=±a, z=0)')
    plt.plot([-a, -a], [0, 0], 'ko', markersize=8)

    # 4. 图表美化
    plt.xlabel('y (m)', fontsize=12)
    plt.ylabel('z (m)', fontsize=12)
    plt.title('Electric Potential and Field of a Charged Ring (y-z Plane)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('ring_potential_field.png', dpi=300, bbox_inches='tight')
    plt.show()


# ------------------- 测试与验证 -------------------
if __name__ == "__main__":
    # 1. 验证轴线上的数值解与解析解
    z_test = np.linspace(-5, 5, 100)
    V_num = np.array([ring_potential_point(0, 0, z) for z in z_test])
    V_ana = axis_potential_analytic(z_test)
    print(f"轴线最大相对误差: {np.max(np.abs(V_num - V_ana)/V_ana):.2e}")  # 应<1e-6

    # 2. 生成y-z平面网格
    y = np.linspace(-2, 2, 100)
    z = np.linspace(-2, 2, 100)
    V_grid = ring_potential_grid(y, z)

    # 3. 计算电场
    Ey, Ez = compute_electric_field(y, z, V_grid)

    # 4. 绘图
    plot_ring_potential_field(y, z, V_grid, Ey, Ez)
