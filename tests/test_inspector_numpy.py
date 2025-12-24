"""Tests for inspector.py with NumPy arrays."""

import numpy as np

from tracerite.inspector import extract_variables, prettyvalue


class TestNumpyArrays:
    """Test handling of actual numpy arrays."""

    def test_numpy_1d_array_small(self):
        """Test pretty printing of small 1D numpy array."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result, fmt = prettyvalue(arr)
        # Should format as comma-separated values (whole numbers shown without decimals)
        assert "1" in result
        assert "5" in result

    def test_numpy_1d_array_large(self):
        """Test pretty printing of large 1D numpy array."""
        arr = np.arange(200, dtype=float)
        result, fmt = prettyvalue(arr)
        # Should show first and last few elements with ellipsis
        assert "…" in result
        assert "0" in result  # Whole numbers shown without decimals

    def test_numpy_2d_array_small(self):
        """Test pretty printing of small 2D numpy array."""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result, fmt = prettyvalue(arr)
        # Should return nested list representation
        assert isinstance(result, list)
        assert len(result) == 2

    def test_numpy_scalar(self):
        """Test pretty printing of numpy scalar."""
        scalar = np.float32(3.14159)
        result, fmt = prettyvalue(scalar)
        # Should format as float
        assert "3.1" in result

    def test_extract_variables_with_numpy_array(self):
        """Test variable extraction with numpy arrays."""
        arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
        variables = {"my_array": arr}
        sourcecode = "my_array"
        rows = extract_variables(variables, sourcecode)

        row_dict = {row[0]: (row[1], row[2]) for row in rows}
        assert "my_array" in row_dict
        typename, value = row_dict["my_array"]
        # Should contain dtype and shape information
        assert "int32" in typename
        assert "2×2" in typename or "2" in typename

    def test_numpy_array_with_different_dtypes(self):
        """Test extraction of arrays with various dtypes."""
        arrays = {
            "float_arr": np.array([1.0, 2.0], dtype=np.float64),
            "int_arr": np.array([1, 2], dtype=np.int64),
            "bool_arr": np.array([True, False], dtype=np.bool_),
        }
        sourcecode = "float_arr + int_arr + bool_arr"
        rows = extract_variables(arrays, sourcecode)

        row_dict = {row[0]: row[1] for row in rows}
        assert "float64" in row_dict["float_arr"]
        assert "int64" in row_dict["int_arr"]
        assert "bool" in row_dict["bool_arr"]
