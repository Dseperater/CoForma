import unittest
import nibabel as nib
import numpy as np
import torch

from coforma.model import FMPNet3D
from coforma.preprocessing import normalize_source, resample_source
from coforma.sliding_window import sliding_window_predict_fields


class InferenceTests(unittest.TestCase):
    def test_normalization(self):
        data = np.random.default_rng(1).uniform(1, 100, (32, 32, 32)).astype(np.float32)
        data[:2] = 0
        normalized = normalize_source(data)
        self.assertTrue(np.isfinite(normalized).all())
        self.assertTrue((normalized[:2] == 0).all())
        self.assertAlmostEqual(float(normalized[data > 0].mean()), 0, places=5)
        with self.assertRaises(ValueError):
            normalize_source(np.zeros_like(data))

    def test_identity_resampling(self):
        data = np.random.default_rng(2).random((16, 24, 32)).astype(np.float32)
        image = nib.Nifti1Image(data, np.diag([1.146, 1.146, 2, 1]))
        np.testing.assert_array_equal(resample_source(image, image), data)

    def test_cpu_forward(self):
        torch.set_num_threads(2)
        model = FMPNet3D(base_channels=8).eval()
        with torch.inference_mode():
            prediction = model(torch.zeros(1, 2, 8, 16, 16))["factor_only_prediction"]
        self.assertEqual(tuple(prediction.shape), (1, 1, 8, 16, 16))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(((prediction >= 0) & (prediction <= 1)).all())

    def test_sliding_window_padding(self):
        class ConstantModel(torch.nn.Module):
            def forward(self, x, coordinate_mode):
                return {"factor_only_prediction": torch.full_like(x[:, :1], 0.5)}
        result = sliding_window_predict_fields(
            ConstantModel(), np.zeros((2, 7, 20, 24), dtype=np.float32),
            ("factor_only_prediction",), (8, 16, 16), 0.5, 2,
            torch.device("cpu"), torch.float32, "drop_s",
        )["factor_only_prediction"]
        self.assertEqual(result.shape, (7, 20, 24))
        np.testing.assert_allclose(result, 0.5)


if __name__ == "__main__":
    unittest.main()
