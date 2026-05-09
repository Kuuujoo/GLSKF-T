from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.io import savemat


SIZE = (120, 120, 30)
TUBAL_RANK = 8
SEEDS = [920, 921, 922]
MISS_LIST = [0.8, 0.9, 0.95]
RESIDUAL_STRENGTH = 0.08
LOWRANK_SMOOTH_SIGMA = (8.0, 0.0, 2.0)
RESIDUAL_SIGMA_SMALL = (0.8, 0.8, 0.4)
RESIDUAL_SIGMA_LARGE = (4.0, 4.0, 1.5)
DATA_VARIANT = "smooth_lowrank_plus_local_detail_residual"


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
    u = gaussian_filter(u, sigma=LOWRANK_SMOOTH_SIGMA, mode="reflect")
    v = gaussian_filter(v, sigma=LOWRANK_SMOOTH_SIGMA, mode="reflect")
    s = fixed_s(TUBAL_RANK, n3)
    x_lowrank_raw = tprod(tprod(u, s), ttranspose(v))
    x_lowrank = 0.20 + 0.60 * normalize01(x_lowrank_raw)
    r_local, residual_energy_ratio = make_local_residual(rng, SIZE, x_lowrank)
    x_raw = x_lowrank + r_local
    clip_ratio = float(np.mean((x_raw < 0.0) | (x_raw > 1.0)))
    x = np.clip(x_raw, 0.0, 1.0)
    return {
        "X": x,
        "X_raw_before_clip": x_raw,
        "X_lowrank": x_lowrank,
        "R_local": r_local,
        "U": u,
        "S": s,
        "V": v,
        "clip_ratio": clip_ratio,
        "residual_energy_ratio": residual_energy_ratio,
    }


def make_local_residual(rng, size, x_lowrank):
    noise = rng.standard_normal(size)
    small = gaussian_filter(noise, sigma=RESIDUAL_SIGMA_SMALL, mode="reflect")
    large = gaussian_filter(noise, sigma=RESIDUAL_SIGMA_LARGE, mode="reflect")
    residual_raw = small - large
    residual_raw = residual_raw - np.mean(residual_raw)

    lowrank_centered = x_lowrank - np.mean(x_lowrank)
    lowrank_norm = np.linalg.norm(lowrank_centered)
    residual_norm = np.linalg.norm(residual_raw)

    if residual_norm <= 0 or lowrank_norm <= 0:
        r_local = np.zeros(size, dtype=np.float64)
        residual_energy_ratio = 0.0
    else:
        r_local = RESIDUAL_STRENGTH * residual_raw / residual_norm * lowrank_norm
        residual_energy_ratio = float(np.linalg.norm(r_local) / lowrank_norm)

    return r_local.astype(np.float64), residual_energy_ratio


def save_case(data_dir, seed, tensor):
    base_name = f"S{seed}"
    common = {
        "X": tensor["X"],
        "X_raw_before_clip": tensor["X_raw_before_clip"],
        "X_lowrank": tensor["X_lowrank"],
        "R_local": tensor["R_local"],
        "U": tensor["U"],
        "S": tensor["S"],
        "V": tensor["V"],
        "seed": seed,
        "tubal_rank": TUBAL_RANK,
        "tensor_size": np.array(SIZE, dtype=np.int32),
        "residual_strength": RESIDUAL_STRENGTH,
        "residual_sigma_small": np.array(RESIDUAL_SIGMA_SMALL, dtype=np.float64),
        "residual_sigma_large": np.array(RESIDUAL_SIGMA_LARGE, dtype=np.float64),
        "clip_ratio": tensor["clip_ratio"],
        "residual_energy_ratio": tensor["residual_energy_ratio"],
        "data_variant": DATA_VARIANT,
    }
    savemat(data_dir / f"{base_name}.mat", common, do_compression=True)
    np.savez_compressed(data_dir / f"{base_name}.npz", **common)

    for missing_rate in MISS_LIST:
        miss_tag = int(round(missing_rate * 100))
        rng = np.random.default_rng(seed * 1000 + miss_tag)
        omega = rng.random(SIZE) > missing_rate
        y = tensor["X"] * omega
        case_name = f"{base_name}_miss{miss_tag}"
        case_data = {
            "X": tensor["X"],
            "X_raw_before_clip": tensor["X_raw_before_clip"],
            "X_lowrank": tensor["X_lowrank"],
            "R_local": tensor["R_local"],
            "Omega": omega,
            "Y": y,
            "seed": seed,
            "missing_rate": missing_rate,
            "tubal_rank": TUBAL_RANK,
            "tensor_size": np.array(SIZE, dtype=np.int32),
            "residual_strength": RESIDUAL_STRENGTH,
            "residual_sigma_small": np.array(RESIDUAL_SIGMA_SMALL, dtype=np.float64),
            "residual_sigma_large": np.array(RESIDUAL_SIGMA_LARGE, dtype=np.float64),
            "clip_ratio": tensor["clip_ratio"],
            "residual_energy_ratio": tensor["residual_energy_ratio"],
            "data_variant": DATA_VARIANT,
        }
        savemat(data_dir / f"{case_name}.mat", case_data, do_compression=True)
        np.savez_compressed(data_dir / f"{case_name}.npz", **case_data)


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        tensor = make_tensor(seed)
        save_case(data_dir, seed, tensor)
        print(
            f"saved S{seed} "
            f"clip_ratio={tensor['clip_ratio']:.6f} "
            f"residual_energy_ratio={tensor['residual_energy_ratio']:.6f}"
        )
        if tensor["clip_ratio"] >= 0.01:
            print(f"warning: S{seed} clip_ratio is {tensor['clip_ratio']:.6f}")


if __name__ == "__main__":
    main()
