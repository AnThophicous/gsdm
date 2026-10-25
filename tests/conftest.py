import numpy as np
import pytest

from gsdm.autograd import default_dtype


@pytest.fixture
def float64_precision():
    with default_dtype(np.float64):
        yield
