import numpy as np
import cmath
from pyhypergeomatrix.hypergeomat import hypergeomPQ

# 1F0 is det(I-X)^(-a)
def test_1F0():
    X = np.array([[3j, 2, 1], [2, 3j, 2], [1, 2, 3j]]) / 100
    x = np.linalg.eigvals(X)
    obtained = hypergeomPQ(15, [4j], [], x)
    expected = np.linalg.det(np.eye(3)-X)**(-4j)
    assert cmath.isclose(obtained, expected, rel_tol=1e-6)