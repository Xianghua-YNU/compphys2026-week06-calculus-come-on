import math

def debye_integrand(x: float) -> float:
    # 处理x趋近于0的极限情况, 避免除以0
    if abs(x) < 1e-12:
        return 0.0
    ex = math.exp(x)
    return (x**4) * ex / ((ex - 1.0) ** 2)

# -------------------------- 改动1: 实现复合梯形积分 --------------------------
def trapezoid_composite(f, a: float, b: float, n: int) -> float:
    """
    复合梯形积分法
    :param f: 被积函数
    :param a: 积分下限
    :param b: 积分上限
    :param n: 分段数
    :return: 积分结果
    """
    # 1. 计算步长h
    h = (b - a) / n
    # 2. 初始化积分和: 首项f(a) + 末项f(b)
    integral_sum = f(a) + f(b)
    # 3. 遍历中间节点, 累加2*f(x_i)
    for i in range(1, n):
        x_i = a + i * h
        integral_sum += 2 * f(x_i)
    # 4. 乘以h/2得到最终结果
    return integral_sum * h / 2

# -------------------------- 改动2: 实现复合Simpson积分 --------------------------
def simpson_composite(f, a: float, b: float, n: int) -> float:
    """
    复合Simpson积分法(要求n为偶数)
    :param f: 被积函数
    :param a: 积分下限
    :param b: 积分上限
    :param n: 分段数(必须为偶数)
    :return: 积分结果
    """
    # 检查n是否为偶数, 不符合则抛出错误
    if n % 2 != 0:
        raise ValueError("Simpson积分要求分段数n必须为偶数!")
    # 1. 计算步长h
    h = (b - a) / n
    # 2. 初始化积分和: 首项f(a) + 末项f(b)
    integral_sum = f(a) + f(b)
    # 3. 遍历奇数节点(i=1,3,5...n-1), 累加4*f(x_i)
    for i in range(1, n, 2):
        x_i = a + i * h
        integral_sum += 4 * f(x_i)
    # 4. 遍历偶数节点(i=2,4,6...n-2), 累加2*f(x_i)
    for i in range(2, n, 2):
        x_i = a + i * h
        integral_sum += 2 * f(x_i)
    # 5. 乘以h/3得到最终结果
    return integral_sum * h / 3

# -------------------------- 改动3: 实现Debye积分 --------------------------
def debye_integral(T: float, theta_d: float = 428.0, method: str = "simpson", n: int = 200) -> float:
    """
    计算Debye积分I(theta_d/T)
    :param T: 温度(开尔文)
    :param theta_d: 德拜温度, 默认428.0(铝的德拜温度)
    :param method: 积分方法, 支持"trapezoid"(梯形法)、"simpson"(Simpson法)
    :param n: 分段数, 默认200
    :return: Debye积分结果
    """
    # 1. 计算积分上限y = theta_d / T
    y = theta_d / T
    # 2. 根据method选择积分方法
    if method.lower() == "trapezoid":
        return trapezoid_composite(debye_integrand, 0.0, y, n)
    elif method.lower() == "simpson":
        return simpson_composite(debye_integrand, 0.0, y, n)
    else:
        raise ValueError(f"不支持的积分方法: {method}, 请选择'trapezoid'或'simpson'")

# -------------------------- 【关键】入口执行逻辑: 脚本运行时自动执行 --------------------------
if __name__ == "__main__":
    # 测试1: 验证梯形法和Simpson法的基础功能(用简单积分测试)
    print("=== 基础积分测试(验证函数正确性) ===")
    # 测试积分: ∫0到1 x² dx = 1/3 ≈ 0.333333
    def f_test(x): return x**2
    trap_test = trapezoid_composite(f_test, 0, 1, 1000)
    simp_test = simpson_composite(f_test, 0, 1, 1000)
    print("∫0^1 x² dx 理论值: 0.333333")
    print(f"梯形法结果: {trap_test:.6f}, 误差: {abs(trap_test - 1/3):.8f}")
    print(f"Simpson法结果: {simp_test:.6f}, 误差: {abs(simp_test - 1/3):.8f}\n")

    # 测试2: Debye积分方法对比(实验要求的误差对比)
    print("=== Debye积分方法对比(T=300K, theta_d=428K) ===")
    T_test = 300.0
    n_list = [10, 20, 50, 100, 200, 500]
    print(f"{'分段数n':<8} {'梯形法结果':<12} {'Simpson法结果':<12} {'差值':<12}")
    for n in n_list:
        trap_res = debye_integral(T_test, method="trapezoid", n=n)
        simp_res = debye_integral(T_test, method="simpson", n=n)
        diff = abs(trap_res - simp_res)
        print(f"{n:<8} {trap_res:<12.6f} {simp_res:<12.6f} {diff:<12.6f}")
