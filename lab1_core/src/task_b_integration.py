import numpy as np

def trapezoid_composite(f, a: float, b: float, n: int) -> float:
    if n < 1:
        raise ValueError("n must be >= 1")
    x = np.linspace(a, b, n+1)
    h = (b - a) / n
    return h * (0.5 * f(x[0]) + np.sum(f(x[1:-1])) + 0.5 * f(x[-1]))

def simpson_composite(f, a: float, b: float, n: int) -> float:
    # 【关键修复】强制偶数分段，否则抛出错误，通过测试校验
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule")
    if n < 2:
        raise ValueError("n must be >= 2")
    x = np.linspace(a, b, n+1)
    h = (b - a) / n
    return h/3 * (f(x[0]) + 4*np.sum(f(x[1:-1:2])) + 2*np.sum(f(x[2:-1:2])) + f(x[-1]))

def debye_integral(T: float, theta_d: float, method: str = 'simpson', n: int = 1000) -> float:
    if T <= 0:
        raise ValueError("T must be positive")
    y = theta_d / T

    def f(x):
        if x == 0:
            return 0.0
        ex = np.exp(x)
        return (x**4 * ex) / (ex - 1)**2

    if method == 'trapezoid':
        return trapezoid_composite(f, 0, y, n)
    elif method == 'simpson':
        return simpson_composite(f, 0, y, n)
    else:
        raise ValueError("method must be 'trapezoid' or 'simpson'")

# 本地测试用例（必须通过）
if __name__ == "__main__":
    # 测试Simpson法偶数分段校验
    try:
        simpson_composite(lambda x: x**2, 0, 1, 3)  # 奇数n，必须报错
        assert False, "Simpson法未校验偶数分段，测试失败"
    except ValueError:
        pass
    # 测试积分精度
    f = lambda x: x**2
    assert abs(simpson_composite(f, 0, 1, 100) - 1/3) < 1e-10
    print("Task B本地测试全部通过！")
