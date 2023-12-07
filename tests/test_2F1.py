import numpy as np
import cmath
from pyhypergeomatrix.hypergeomat import hypergeomPQ

# Herz's relation for 2F1
def test_2F1():
    X = np.array([[3j, 2, 1], [2, 3j, 2], [1, 2, 3j]]) / 100
    x = np.linalg.eigvals(X)
    o1 = hypergeomPQ(15, [1,2j], [3], x)
    x = np.linalg.eigvals(-X @ np.linalg.inv(np.eye(3)-X))
    o2 = np.linalg.det(np.eye(3)-X)**(-2j) * hypergeomPQ(15, [3-1,2j], [3], x)
    assert cmath.isclose(o1, o2, rel_tol=1e-6)