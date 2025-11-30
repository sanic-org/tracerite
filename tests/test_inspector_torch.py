"""Tests for inspector.py with PyTorch tensors."""

import pytest
import torch

from tracerite.inspector import extract_variables, prettyvalue


class TestPyTorchTensors:
    """Test handling of actual PyTorch tensors."""

    def test_torch_1d_tensor_small(self):
        """Test pretty printing of small 1D tensor."""
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = prettyvalue(tensor)
        # Should format as comma-separated values
        assert "1.00" in result
        assert "5.00" in result

    def test_torch_1d_tensor_large(self):
        """Test pretty printing of large 1D tensor."""
        tensor = torch.arange(200, dtype=torch.float32)
        result = prettyvalue(tensor)
        # Should show first and last few elements with ellipsis
        assert "…" in result

    def test_torch_2d_tensor_small(self):
        """Test pretty printing of small 2D tensor."""
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = prettyvalue(tensor)
        # Should return nested list representation
        assert isinstance(result, list)
        assert len(result) == 2

    def test_extract_variables_with_torch_tensor(self):
        """Test variable extraction with torch tensors."""
        tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        variables = {"my_tensor": tensor}
        sourcecode = "my_tensor"
        rows = extract_variables(variables, sourcecode)

        row_dict = {row[0]: (row[1], row[2]) for row in rows}
        assert "my_tensor" in row_dict
        typename, value = row_dict["my_tensor"]
        # Should contain dtype and shape information
        assert "int32" in typename
        assert "2×2" in typename or "2" in typename

    def test_torch_tensor_cpu_device(self):
        """Test that CPU device is not shown in typename."""
        tensor = torch.tensor([1.0, 2.0], device="cpu")
        variables = {"cpu_tensor": tensor}
        sourcecode = "cpu_tensor"
        rows = extract_variables(variables, sourcecode)

        row_dict = {row[0]: row[1] for row in rows}
        typename = row_dict["cpu_tensor"]
        # CPU device should not be shown
        assert "@" not in typename or "@cpu" not in typename

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_torch_tensor_cuda_device(self):
        """Test that CUDA device is shown in typename."""
        tensor = torch.tensor([1.0, 2.0], device="cuda")
        variables = {"gpu_tensor": tensor}
        sourcecode = "gpu_tensor"
        rows = extract_variables(variables, sourcecode)

        row_dict = {row[0]: row[1] for row in rows}
        typename = row_dict["gpu_tensor"]
        # CUDA device should be shown
        assert "@cuda" in typename or "cuda" in typename.lower()

    def test_torch_different_dtypes(self):
        """Test extraction of tensors with various dtypes."""
        tensors = {
            "float_tensor": torch.tensor([1.0, 2.0], dtype=torch.float32),
            "int_tensor": torch.tensor([1, 2], dtype=torch.int64),
            "bool_tensor": torch.tensor([True, False], dtype=torch.bool),
        }
        sourcecode = "float_tensor + int_tensor + bool_tensor"
        rows = extract_variables(tensors, sourcecode)

        row_dict = {row[0]: row[1] for row in rows}
        assert "float32" in row_dict["float_tensor"]
        assert "int64" in row_dict["int_tensor"]
        assert "bool" in row_dict["bool_tensor"]
