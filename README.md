# CoForma
Learning to Synthesize Susceptibility-Weighted MRI through Coupled Formation Paths

Inference code and pretrained weights for synthesizing susceptibility-weighted
MRI (SWI) from paired T1w and T2w volumes. The released model uses the fixed
factor-only (drop-s) path. Training code and imaging data are not included.

## Installation

Tested on Linux with Python 3.10, PyTorch 2.4.1 (CUDA 12.1), and an NVIDIA GPU.

```bash
git clone https://github.com/Dseperater/CoForma.git
cd CoForma
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python download_weights.py
```

For CPU-only use, replace `cu121` with `cpu` in the PyTorch installation command.
CPU inference is slower and uses float32.

## Run inference

Inputs must be **spatially registered** T1w/T2w NIfTI volumes. `--reference`
defines the output shape and affine; its voxel intensities are not read.
This script resamples already registered images; it does not perform registration.

```bash
python infer.py \
  --t1 /path/to/registered_T1w.nii.gz \
  --t2 /path/to/registered_T2w.nii.gz \
  --reference /path/to/output_grid.nii.gz \
  --checkpoint weights/coforma.pt \
  --output /path/to/synthesized_SWI.nii.gz
```

The study uses an axial RAS-oriented SWI grid (approximately
1.146 × 1.146 × 2.0 mm). Supply a matching registered grid for study-like inference;
a blank NIfTI carrying the required shape and affine is sufficient. No measured
SWI, GRE magnitude, phase, or target-derived brain mask is needed as a model input.
Arbitrary acquisition protocols or grids have not been validated.

Each input is linearly resampled onto the reference grid and independently
normalized over its finite positive support: percentile clipping at 0.5/99.5,
z-score normalization, and clipping to [-5, 5]. Other voxels are set to zero.
Normalization does not use the target. The third NIfTI axis is the depth axis.

Inference uses 48 × 128 × 128 patches (D × H × W), 50% overlap, Gaussian blending,
and EMA weights. The output is **normalized SWI in [0, 1]**, not raw scanner
intensity. Its shape and affine match the reference. Existing output files are
not overwritten.

The default GPU precision is bfloat16. Use `--precision float32` on GPUs without
bfloat16 support, `--device cpu` for CPU inference, or `--batch-size 1` to reduce
GPU memory usage. Different precision or hardware can produce small numerical
differences.

## Weights and checks

Weights are hosted in [Releases](https://github.com/Dseperater/CoForma/releases/tag/v1.0.0).
The download script verifies SHA-256 before installing them. The checkpoint is
a tensor-only EMA state dictionary, loaded with `weights_only=True`; it contains
no optimizer state, training records, or subject data.

```bash
python -m unittest discover -s tests
```

The release was checked against the original frozen implementation on one
complete held-out volume using the same bfloat16 sliding-window settings:
maximum absolute output difference was 0.0. Input normalization was also
verified to match exactly. These checks establish implementation consistency,
not validation on new acquisition protocols.

This model is for research, not clinical diagnosis.
