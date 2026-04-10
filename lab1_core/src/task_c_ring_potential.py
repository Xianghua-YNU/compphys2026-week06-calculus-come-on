import math
import numpy as np

def ring_potential_point(x, y, z, a=1.0, q=1.0, n_phi=720):
    dphi = 2 * math.pi / n_phi
    v_sum = 0.0

    for i in range(n_phi):
        phi = i * dphi
        dx = x - a * math.cos(phi)
        dy = y - a * math.sin(phi)
        r_sq = dx**2 + dy**2 + z**2

        # ===================== 老师要求的奇点处理 =====================
        if r_sq < 1e-12:
            inv_r = 0.0
        else:
            inv_r = 1.0 / math.sqrt(r_sq)
        # ==============================================================

        v_sum += inv_r

    V = q * v_sum * dphi / (2 * math.pi)
    return V

def axis_potential_analytic(z, a=1.0, q=1.0):
    return q / (2 * math.sqrt(a**2 + z**2))

def ring_potential_grid(ys, zs, x0=0.0, a=1.0, q=1.0, n_phi=360):
    ny = len(ys)
    nz = len(zs)
    V = np.zeros((nz, ny))

    for i in range(nz):
        z = zs[i]
        for j in range(ny):
            y = ys[j]
            V[i, j] = ring_potential_point(x0, y, z, a, q, n_phi)

    return V
