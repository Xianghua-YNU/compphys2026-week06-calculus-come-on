import numpy as np
import matplotlib.pyplot as plt

# 严格匹配初始函数签名，无任何修改
def ring_potential_point(x: float, y: float, z: float, a: float = 1.0, q: float = 1.0) -> float:
    # 纯numpy实现，无scipy依赖，通过测试
    n_phi = 2000
    phi = np.linspace(0, 2 * np.pi, n_phi + 1)
    dphi = 2 * np.pi / n_phi

    dx = x - a * np.cos(phi)
    dy = y - a * np.sin(phi)
    r_sq = dx**2 + dy**2 + z**2
    r_sq = np.maximum(r_sq, 1e-20)  # 数值稳定，避免除以0
    integrand = 1.0 / np.sqrt(r_sq)

    # 复合Simpson法积分
    integral = dphi / 3 * (
        integrand[0]
        + 4 * np.sum(integrand[1:-1:2])
        + 2 * np.sum(integrand[2:-1:2])
        + integrand[-1]
    )
    return q / (2 * np.pi) * integral

def ring_potential_grid(y_grid: np.ndarray, z_grid: np.ndarray, x0: float = 0.0,
                        a: float = 1.0, q: float = 1.0) -> np.ndarray:
    # 严格匹配输入输出，返回(Nz, Ny)矩阵
    y_grid = np.asarray(y_grid)
    z_grid = np.asarray(z_grid)
    Ny, Nz = len(y_grid), len(z_grid)
    V = np.zeros((Nz, Ny), dtype=np.float64)

    for i in range(Nz):
        z = z_grid[i]
        for j in range(Ny):
            y = y_grid[j]
            V[i, j] = ring_potential_point(x0, y, z, a, q)
    return V

def axis_potential_analytic(z: float, a: float = 1.0, q: float = 1.0) -> float:
    return q / np.sqrt(a**2 + z**2)

# 本地测试用例（必须通过）
if __name__ == "__main__":
    # 测试轴线精度
    z_test = np.array([0.0, 1.0, 2.0])
    for z in z_test:
        v_num = ring_potential_point(0, 0, z)
        v_ana = axis_potential_analytic(z)
        assert abs(v_num - v_ana) < 1e-6, f"z={z} 轴线测试失败"
    # 测试网格形状
    y = np.linspace(-2,2,50)
    z = np.linspace(-2,2,50)
    V = ring_potential_grid(y, z)
    assert V.shape == (50,50), f"网格形状错误，预期(50,50)，实际{V.shape}"
    print("Task C本地测试全部通过！")
