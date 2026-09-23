"""Synthesize normalized SWI from registered T1w/T2w NIfTI volumes."""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from coforma.model import FMPNet3D
from coforma.preprocessing import normalize_source, resample_source
from coforma.sliding_window import sliding_window_predict_fields


def load_model(checkpoint, device):
    model = FMPNet3D(coordinate_maximum=1.38629436112).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()


def predict(model, source, device, batch_size=2, precision="bfloat16"):
    if not np.isfinite(source).all():
        raise ValueError("Non-finite normalized inputs.")
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32
    prediction = sliding_window_predict_fields(
        model, np.ascontiguousarray(source, dtype=np.float32),
        output_keys=("factor_only_prediction",),
        patch_size=(48, 128, 128), overlap=0.5,
        sw_batch_size=batch_size, device=device, amp_dtype=dtype,
        coordinate_mode="drop_s",
    )["factor_only_prediction"]
    if not np.isfinite(prediction).all():
        raise RuntimeError("Non-finite prediction.")
    return prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1", type=Path, required=True)
    parser.add_argument("--t2", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True,
                        help="Registered output-grid NIfTI; only shape and affine are used.")
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/coforma.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new output filename.")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    device = torch.device(args.device)
    if device.type == "cuda" and args.precision == "bfloat16":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                parser.error("GPU does not support bfloat16; use --precision float32.")
    torch.manual_seed(2026)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    reference = nib.load(args.reference)
    source_hwd = np.stack([
        normalize_source(resample_source(nib.load(path), reference))
        for path in (args.t1, args.t2)
    ])
    source = source_hwd.transpose(0, 3, 1, 2).copy()
    model = load_model(args.checkpoint, device)
    prediction = predict(model, source, device, args.batch_size, args.precision)
    result = nib.Nifti1Image(prediction.transpose(1, 2, 0), reference.affine)
    result.header.set_xyzt_units(reference.header.get_xyzt_units()[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(result, args.output)
    print(f"Saved normalized SWI: {args.output}")


if __name__ == "__main__":
    main()
