"""Independent source normalization and affine-grid resampling."""
import numpy as np
from scipy.ndimage import affine_transform


def normalize_source(data):
    data = np.asarray(data, dtype=np.float32)
    support = np.isfinite(data) & (data > 0)
    if int(support.sum()) < 10000:
        raise ValueError("Input has fewer than 10,000 positive finite voxels.")
    low, high = np.percentile(data[support], (0.5, 99.5))
    clipped = np.clip(data, float(low), float(high))
    mean = float(clipped[support].mean())
    std = float(clipped[support].std())
    if not np.isfinite(std) or std <= 1e-6:
        raise ValueError("Degenerate input intensity distribution.")
    output = np.zeros(data.shape, dtype=np.float32)
    output[support] = (clipped[support] - mean) / std
    return np.clip(output, -5.0, 5.0).astype(np.float32)


def resample_source(image, reference):
    if len(image.shape) != 3 or len(reference.shape) != 3:
        raise ValueError("Inputs and reference must be 3D NIfTI volumes.")
    transform = np.linalg.inv(image.affine) @ reference.affine
    return affine_transform(
        np.asarray(image.dataobj, dtype=np.float32),
        matrix=transform[:3, :3].astype(np.float64),
        offset=transform[:3, 3].astype(np.float64),
        output_shape=reference.shape,
        order=1, mode="constant", cval=0.0, prefilter=False,
    ).astype(np.float32)
