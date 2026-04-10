import numpy as np

def ring_potential_point(x: float, y: float, z: float, a: float = 1.0, q: float = 1.0, n_phi: int = 720) -> float:
    # 离散积分：phi从0到2π，梯形法近似
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    dx = x - a * np.cos(phi)
    dy = y - a * np.sin(phi)
    r = np.sqrt(dx**2 + dy**2 + z**2)
    # 避免r=0奇点，截断到1e-10
    r[r < 1e-10] = 1e-10
    integrand = 1.0 / r
    # 梯形法积分
    integral = np.sum(integrand) * dphi
    return (q / (2 * np.pi)) * integral

def ring_potential_grid(y_grid, z_grid, x0: float = 0.0, a: float = 1.0, q: float = 1.0, n_phi: int = 720):
    ny = len(y_grid)
    nz = len(z_grid)
    V = np.zeros((nz, ny))
    for i in range(nz):
        z = z_grid[i]
        for j in range(ny):
            y = y_grid[j]
            V[i, j] = ring_potential_point(x0, y, z, a, q, n_phi)
    return V

def axis_potential_analytic(z: float, a: float = 1.0, q: float = 1.0) -> float:
    return q / np.sqrt(a * a + z * z)
