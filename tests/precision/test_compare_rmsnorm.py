"""Validate RMSNorm analytic gradients independently of mini-infer."""

import unittest

import torch

from compare_rmsnorm import analytic


class RMSNormReferenceTest(unittest.TestCase):
    def test_hand_calculated_values_and_gradients(self):
        x = torch.tensor([[1., -1.], [1., -1.]], dtype=torch.float64)
        w = torch.tensor([2., -1.], dtype=torch.float64)
        g = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
        result = analytic(x, w, g, 3.)
        expected = {"forward": [[1., .5], [1., .5]],
                    "grad_x": [[.75, -.75], [2.375, -1.375]],
                    "grad_weight": [2., -3.]}
        for key, values in expected.items():
            torch.testing.assert_close(result[key], torch.tensor(values, dtype=torch.float64),
                                       rtol=0, atol=0)

    def test_analytic_gradients_match_central_differences(self):
        for shape in ((3,), (2, 3), (2, 2, 3), (2, 1)):
            count = 1
            for size in shape:
                count *= size
            x = torch.linspace(-1.5, 2., count, dtype=torch.float64).reshape(shape)
            w = torch.linspace(-.75, 1.25, shape[-1], dtype=torch.float64)
            g = torch.linspace(-.5, 1., count, dtype=torch.float64).reshape(shape)
            result = analytic(x, w, g, .1)
            step = 1e-5
            def loss(a, b):
                return (torch.nn.functional.rms_norm(a, (shape[-1],), b, .1) * g).sum()
            for parameter, key in ((x, "grad_x"), (w, "grad_weight")):
                numerical = torch.empty_like(parameter)
                for i in range(parameter.numel()):
                    original = parameter.flatten()[i].item()
                    parameter.flatten()[i] = original + step
                    plus = loss(x, w)
                    parameter.flatten()[i] = original - step
                    minus = loss(x, w)
                    parameter.flatten()[i] = original
                    numerical.flatten()[i] = (plus - minus) / (2 * step)
                torch.testing.assert_close(result[key], numerical, rtol=1e-7, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
