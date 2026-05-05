from pathlib import Path

import numpy as np
from scipy.io import savemat


SIZE = (200, 200, 30)
TUBAL_RANK = 8
SEEDS = [920, 921, 922]
MISS_LIST = [0.8, 0.9, 0.95]


def tprod(a, b):
    a_hat = np.fft.fft(a, axis=2)
    b_hat = np.fft.fft(b, axis=2)
    c_hat = np.zeros((a.shape[0], b.shape[1], a.shape[2]), dtype=np.complex128)
    for k in range(a.shape[2]):
        c_hat[:, :, k] = a_hat[:, :, k] @ b_hat[:, :, k]
    return np.fft.ifft(c_hat, axis=2).real


def ttranspose(a):
    n1, n2, n3 = a.shape
    out = np.zeros((n2, n1, n3), dtype=a.dtype)
    out[:, :, 0] = a[:, :, 0].T
    for k in range(1, n3):
        out[:, :, k] = a[:, :, n3 - k].T
    return out


def fixed_s(rank, n3):
    s = np.zeros((rank, rank, n3), dtype=np.float64)
    t = np.arange(n3)
    for i in range(rank):
        base = 1.0 / (1.0 + 0.18 * i)
        tube = base * (1.0 + 0.12 * np.cos(2 * np.pi * (i + 1) * t / n3))
        s[i, i, :] = tube
    return s


def normalize01(x):
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax <= xmin:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def make_tensor(seed):
    n1, n2, n3 = SIZE
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n1, TUBAL_RANK, n3))
    v = rng.standard_normal((n2, TUBAL_RANK, n3))
    s = fixed_s(TUBAL_RANK, n3)
    x = tprod(tprod(u, s), ttranspose(v))
    x = normalize01(x)
    return x, u, s, v


def save_case(data_dir, seed, x, u, s, v):
    base_name = f"S{seed}"
    common = {
        "X": x,
        "U": u,
        "S": s,
        "V": v,
        "seed": seed,
        "tubal_rank": TUBAL_RANK,
        "tensor_size": np.array(SIZE, dtype=np.int32),
    }
    savemat(data_dir / f"{base_name}.mat", common, do_compression=True)
    np.savez_compressed(data_dir / f"{base_name}.npz", **common)

    for missing_rate in MISS_LIST:
        miss_tag = int(round(missing_rate * 100))
        rng = np.random.default_rng(seed * 1000 + miss_tag)
        omega = rng.random(SIZE) > missing_rate
        y = x * omega
        case_name = f"{base_name}_miss{miss_tag}"
        case_data = {
            "X": x,
            "Omega": omega,
            "Y": y,
            "seed": seed,
            "missing_rate": missing_rate,
            "tubal_rank": TUBAL_RANK,
            "tensor_size": np.array(SIZE, dtype=np.int32),
        }
        savemat(data_dir / f"{case_name}.mat", case_data, do_compression=True)
        np.savez_compressed(data_dir / f"{case_name}.npz", **case_data)


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        x, u, s, v = make_tensor(seed)
        save_case(data_dir, seed, x, u, s, v)
        print(f"saved S{seed}")


if __name__ == "__main__":
    main()
