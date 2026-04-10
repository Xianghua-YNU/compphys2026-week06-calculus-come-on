import math

def debye_integrand(x: float) -> float:
    """
    Debye积分被积函数: x^4 * e^x / (e^x - 1)^2
    处理x→0的极限情况，避免除以0
    """
    if abs(x) < 1e-12:
        return 0.0
    ex = math.exp(x)
    return (x**4) * ex / ((ex - 1.0) ** 2)

def trapezoid_composite(f, a: float, b: float, n: int) -> float:
    """
    复合梯形积分法
    :param f: 被积函数
    :param a: 积分下限
    :param b: 积分上限
    :param n: 分段数
    :return: 积分结果
    """
    h = (b - a) / n
    integral_sum = f(a) + f(b)
    for i in range(1, n):
        x_i = a + i * h
        integral_sum += 2 * f(x_i)
    return integral_sum * h / 2

def simpson_composite(f, a: float, b: float, n: int) -> float:
    """
    复合Simpson积分法（要求n为偶数）
    :param f: 被积函数
    :param a: 积分下限
    :param b: 积分上限
    :param n: 分段数（必须为偶数）
    :return: 积分结果
    """
    if n % 2 != 0:
        raise ValueError("simpson_composite 要求n为偶数")
    h = (b - a) / n
    integral_sum = f(a) + f(b)
    for i in range(1, n, 2):
        x_i = a + i * h
        integral_sum += 4 * f(x_i)
    for i in range(2, n, 2):
        x_i = a + i * h
        integral_sum += 2 * f(x_i)
    return integral_sum * h / 3

def debye_integral(T: float, theta_d: float = 428.0, method: str = "simpson", n: int = 200) -> float:
    """
    计算Debye积分I(theta_d / T)
    :param T: 温度（开尔文）
    :param theta_d: 德拜温度，默认428.0
    :param method: 积分方法，支持"trapezoid"、"simpson"
    :param n: 分段数，默认200
    :return: Debye积分结果
    """
    y = theta_d / T
    if method.lower() == "trapezoid":
        return trapezoid_composite(debye_integrand, 0.0, y, n)
    elif method.lower() == "simpson":
        return simpson_composite(debye_integrand, 0.0, y, n)
    else:
        raise ValueError(f"不支持的积分方法: {method}")
