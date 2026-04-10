import math

def rate_3alpha(T: float) -> float:
    """
    计算3-α反应率q(T)
    q(T) = 5.09e11 * T8^(-3) * exp(-44.027 / T8), 其中T8 = T / 1e8
    """
    T8 = T / 1e8
    return 5.09e11 * (T8 ** (-3)) * math.exp(-44.027 / T8)

def finite_diff_dq_dt(T0: float, h: float = 1e-8) -> float:
    """
    前向差分近似计算dq/dT在T0处的值
    dq/dT ≈ [q(T0 + ΔT) - q(T0)] / ΔT, 其中ΔT = h * T0
    """
    delta_T = h * T0
    q0 = rate_3alpha(T0)
    q1 = rate_3alpha(T0 + delta_T)
    return (q1 - q0) / delta_T

def sensitivity_nu(T0: float, h: float = 1e-8) -> float:
    """
    计算温度敏感性指数ν
    ν = (T / q * dq/dT) 在T0处的值
    """
    q0 = rate_3alpha(T0)
    dqdt = finite_diff_dq_dt(T0, h)
    return (T0 / q0) * dqdt

def nu_table(T_values: list, h: float = 1e-8) -> list:
    """
    生成温度-敏感性指数表格
    输入: T_values = [T1, T2, ...]
    输出: [[T1, nu1], [T2, nu2], ...]
    """
    table = []
    for T in T_values:
        nu = sensitivity_nu(T, h)
        table.append([T, nu])
    return table
