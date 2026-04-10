import math

def debye_integrand(x: float) -> float:
    if abs(x) < 1e-12:
        return 0.0
    ex = math.exp(x)
    return (x**4) * ex / ((ex - 1.0) ** 2)

def trapezoid_composite(f, a: float, b: float, n: int) -> float:
    if n <= 0:
        raise ValueError("n must be a positive integer")
    h = (b - a) / n
    total = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        x = a + i * h
        total += f(x)
    return total * h

def simpson_composite(f, a: float, b: float, n: int) -> float:
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule")
    if n <= 0:
        raise ValueError("n must be a positive integer")
    h = (b - a) / n
    total = f(a) + f(b)
    # 奇数索引项：4倍权重
    for i in range(1, n, 2):
        x = a + i * h
        total += 4 * f(x)
    # 偶数索引项：2倍权重
    for i in range(2, n, 2):
        x = a + i * h
        total += 2 * f(x)
    return total * h / 3.0

def debye_integral(T: float, theta_d: float = 428.0, method: str = "simpson", n: int = 200) -> float:
    if T <= 0:
        raise ValueError("Temperature T must be positive")
    y = theta_d / T
    if method.lower() == "trapezoid":
        return trapezoid_composite(debye_integrand, 0.0, y, n)
    elif method.lower() == "simpson":
        return simpson_composite(debye_integrand, 0.0, y, n)
    else:
        raise ValueError(f"Unknown method: {method}, choose 'trapezoid' or 'simpson'")

# 自测代码（误差对比）
if __name__ == "__main__":
    # 测试：T=300K，theta_d=428K，y≈1.4267
    T_test = 300.0
    n_list = [10, 20, 50, 100, 200, 500]
    print("Task B 积分方法对比：")
    print(f"{'n':<4} | {'梯形法结果':<10} | {'Simpson法结果':<12}")
    print("-" * 35)
    for n in n_list:
        trap = debye_integral(T_test, method="trapezoid", n=n)
        simp = debye_integral(T_test, method="simpson", n=n)
        print(f"{n:<4} | {trap:<10.6f} | {simp:<12.6f}")
