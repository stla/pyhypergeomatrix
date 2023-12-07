import numpy as np
import cmath
from pyhypergeomatrix.hypergeomat import hypergeomPQ

# 0F0 is the exponential of the trace
def test_0F0():
    X = np.array([[3j, 2, 1], [2, 3j, 2], [1, 2, 3j]]) / 10
    x = np.linalg.eigvals(X)
    obtained = hypergeomPQ(10, [], [], x)
    expected = cmath.exp(np.sum(np.diag(X)))
    assert cmath.isclose(obtained, expected, rel_tol=1e-6)