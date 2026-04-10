import numpy as np

def rate_3alpha(T: float) -> float:
    T8 = T / 1.0e8
    return 5.09e11 * (T8 ** (-3.0)) * np.exp(-44.027 / T8)

def finite_diff_dq_dT(T0: float, h: float = 1e-8) -> float:
    # 严格按照要求：ΔT = h * T0，避免将h作为绝对增量
    delta_T = h * T0
    q0 = rate_3alpha(T0)
    q1 = rate_3alpha(T0 + delta_T)
    return (q1 - q0) / delta_T

def sensitivity_nu(T0: float, h: float = 1e-8) -> float:
    q0 = rate_3alpha(T0)
    dq_dT = finite_diff_dq_dT(T0, h)
    # 严格使用q(T0)，避免误写为q(T0+ΔT)
    return (T0 / q0) * dq_dT

def nu_table(T_values, h: float = 1e-8):
    result = []
    for T in T_values:
        nu = sensitivity_nu(T, h)
        result.append((T, nu))
    return result

# 自测代码（直接运行验证结果）
if __name__ == "__main__":
    test_T = [1.0e8, 2.5e8, 5.0e8, 1.0e9, 2.5e9, 5.0e9]
    table = nu_table(test_T)
    print("Task A 计算结果：")
    for T, nu in table:
        print(f"{T:.2e} K : nu = {nu:.2f}")
