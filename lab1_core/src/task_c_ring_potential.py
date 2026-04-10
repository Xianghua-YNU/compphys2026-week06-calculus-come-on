import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 1. 单点电势计算
def ring_potential_point(x, y, z, a=1.0, q=1.0):
    def integrand(phi):
        dx = x - a * np.cos(phi)
        dy = y - a * np.sin(phi)
        r = np.sqrt(dx**2 + dy**2 + z**2)
        # 避免圆环上r=0的奇点
        if r < 1e-10:
            return 0.0
        return 1.0 / r
    # 高精度数值积分
    integral, _ = quad(integrand, 0, 2 * np.pi)
    return (q / (2 * np.pi)) * integral

# 2. yz平面网格电势计算（x=0）
def ring_potential_grid(y_grid, z_grid, x0=0.0, a=1.0, q=1.0):
    ny = len(y_grid)
    nz = len(z_grid)
    V = np.zeros((nz, ny))
    for i in range(nz):
        z = z_grid[i]
        for j in range(ny):
            y = y_grid[j]
            V[i, j] = ring_potential_point(x0, y, z, a, q)
    return V

# 3. 电场计算（数值梯度，E = -∇V）
def compute_electric_field(V, y_grid, z_grid):
    dy = y_grid[1] - y_grid[0]
    dz = z_grid[1] - z_grid[0]
    # 中心差分计算梯度
    Ey, Ez = np.gradient(-V, dy, dz)
    return Ey, Ez

# 4. 可视化：等势线 + 电场矢量
def plot_ring_field():
    # 生成yz平面网格
    y = np.linspace(-2.0, 2.0, 100)
    z = np.linspace(-2.0, 2.0, 100)
    Y, Z = np.meshgrid(y, z)
    
    # 计算电势
    V = ring_potential_grid(y, z)
    
    # 计算电场并归一化（保证箭头大小一致）
    Ey, Ez = compute_electric_field(V, y, z)
    E_mag = np.sqrt(Ey**2 + Ez**2)
    Ey_norm = Ey / (E_mag + 1e-10)
    Ez_norm = Ez / (E_mag + 1e-10)
    
    # 绘图
    plt.figure(figsize=(8, 6))
    # 等势线
    contour = plt.contour(Y, Z, V, levels=20, cmap="viridis")
    plt.colorbar(contour, label="Electric Potential (V)")
    # 电场矢量（quiver，降采样避免箭头重叠）
    plt.quiver(Y[::5, ::5], Z[::5, ::5], Ey_norm[::5, ::5], Ez_norm[::5, ::5],
               color="red", scale=30, width=0.003, alpha=0.8, label="Electric Field")
    # 绘制圆环轮廓
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), "k--", linewidth=2, label="Charged Ring (a=1, q=1)")
    
    plt.xlabel("y (m)")
    plt.ylabel("z (m)")
    plt.title("Electric Potential and Field of a Charged Ring (yz-plane, x=0)")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("ring_potential_field.png", dpi=300, bbox_inches="tight")
    plt.show()

# 自测运行
if __name__ == "__main__":
    # 验证原点电势（理论值q/a=1.0）
    V_origin = ring_potential_point(0, 0, 0)
    print(f"Task C 原点电势验证：{V_origin:.4f}（理论值：1.0000）")
    # 绘制场图
    plot_ring_field()
