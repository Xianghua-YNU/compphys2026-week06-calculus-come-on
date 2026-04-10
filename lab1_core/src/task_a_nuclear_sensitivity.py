import numpy as np

def rate_3alpha(T: float) -> float:
    # 严格按题目公式，T8 = T / 1e8（绝对不能写10^8！）
    T8 = T / 1e8
    return 5.09e11 * (T8 ** (-3)) * np.exp(-44.027 / T8)

def finite_diff_dq_dT(T0: float, h: float = 1e-8) -> float:
    # 严格按前向差分，ΔT = h*T0（绝对不能写ΔT=h）
    dT = h * T0
    q0 = rate_3alpha(T0)
    q1 = rate_3alpha(T0 + dT)
    return (q1 - q0) / dT

def sensitivity_nu(T0: float, h: float = 1e-8) -> float:
    # 严格按定义，用q(T0)，不是q(T0+dT)
    q0 = rate_3alpha(T0)
    dq_dT = finite_diff_dq_dT(T0, h)
    return (T0 / q0) * dq_dT

def nu_table(T_values: np.ndarray, h: float = 1e-8) -> list:
    res = []
    for T in T_values:
        nu = sensitivity_nu(T, h)
        res.append([T, nu])
    return res

# 本地测试用例（必须通过，否则CI必挂）
if __name__ == "__main__":
    T_list = np.array([1.0e8, 2.5e8, 5.0e8, 1.0e9, 2.5e9, 5.0e9])
    h = 1e-8
    # 验证1e8K时nu≈41.03（题目给的参考值）
    nu_1e8 = sensitivity_nu(1.0e8, h)
    assert abs(nu_1e8 - 41.03) < 0.1, f"1e8K nu错误，预期41.03，实际{nu_1e8:.2f}"
    print("Task A本地测试全部通过！")
