"""Peak signal-to-noise ratio and mean squared error."""


def mse(a, b):
    n = max(1, len(a))
    return sum(sum((a[i][j] - b[i][j]) ** 2 for j in range(len(a[i]))) for i in range(n)) / n


def psnr(a, b, max_val=1.0):
    e = mse(a, b)
    if e <= 1e-12:
        return float("inf")
    import math
    return 20 * math.log10(max_val) - 10 * math.log10(e)
