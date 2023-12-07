from math import isinf
import numpy as np


def __DictParts(m, n):
    D = []
    Last = np.array([[0], [m], [m]])
    end = 0
    for i in range(n):
        NewLast = np.empty((3, 0), dtype=int)
        ncol = Last.shape[1]
        for j in range(ncol):
            record = Last[:, j]
            lack = record[1]
            l = min(lack, record[2])
            if l > 0 :
                D.append((record[0]+1, end+1))
                x = np.empty((3, l), dtype=int)
                for k in range(l):
                    x[:, k] = np.array([end+k+1, lack-k-1, k+1])
                NewLast = np.hstack((NewLast, x))
                end += l
        Last = NewLast
    return (dict(D), end)


def _N(dico, kappa):
    kappa = kappa[kappa > 0]
    l = len(kappa)
    if l == 0:
        return 1
    return dico[_N(dico, kappa[:(l-1)])] + kappa[-1] 


def _T(alpha, a, b, kappa):
    i = len(kappa)
    if i == 0 or kappa[0] == 0:
        return 1
    c = kappa[-1] - 1 - (i - 1) / alpha
    d = alpha * kappa[-1] - i
    s = np.asarray(range(1, kappa[-1]), dtype=int)
    e = d - alpha * s + np.array(
        list(map(lambda j: np.count_nonzero(kappa >= j), s))
    )
    g = e + 1
    ss = range(i-1)
    f = alpha * kappa[ss] - (np.asarray(ss) + 1) - d
    h = alpha + f
    l = h * f
    prod1 = np.prod(a + c)
    prod2 = np.prod(b + c)
    prod3 = np.prod((g - alpha) * e / (g * (e + alpha)))
    prod4 = np.prod((l - f) / (l + h))
    out = prod1 / prod2 * prod3 * prod4
    return 0 if isinf(out) or np.isnan(out) else out


def __dualPartition(kappa):
    out = []
    if len(kappa) > 0 and kappa[0] > 0:
        for i in range(1, kappa[0]+1):
            out.append(np.count_nonzero(kappa >= i))
    return np.asarray(out, dtype=int)


def __betaratio(kappa, mu, k, alpha):
    muk = mu[k-1]
    t = k - alpha * muk
    prod1 = prod2 = prod3 = 1
    if k > 0:
        u = np.array(
            list(map(lambda i: t + 1 - i + alpha * kappa[i-1], range(1,k+1)))
        )
        prod1 = np.prod(u / (u + alpha - 1))
    if k > 1:
        v = np.array(
            list(map(lambda i: t - i + alpha * mu[i-1], range(1, k)))
        )
        prod2 = np.prod((v + alpha) / v)
    if muk > 1:
        muPrime = __dualPartition(mu)
        w = np.array(
            list(map(lambda i: muPrime[i-1] - t - alpha * i, range(1, muk)))
        )
        prod3 = np.prod((w + alpha) / w)
    return alpha * prod1 * prod2 * prod3


def __hypergeomI(m, alpha, a, b, n, x):
    def summation(i, z, j, kappa):
        def go(kappai, zz, s):
            if i == 0 and kappai > j or i > 0 and kappai > min(kappa[i-1], j):
                return s
            kappap = np.vstack((kappa, [kappai]))
            t = _T(alpha, a, b, kappap[kappap > 0])
            zp = zz * x * (n - i + alpha * (kappai -1)) * t
            sp = s
            if j > kappai and i <= n:
                sp += summation(i+1, zp, j - kappai, kappap)
            spp = sp + zp
            return go(kappai + 1, zp, spp)
        return go(1, z, 0)
    return 1 + summation(0, 1, m, np.empty((0,1), dtype=int))


def hypergeomPQ(m, a, b, x, alpha=2):
    """
    Hypergeometric function of a matrix argument.

    Parameters
    ----------
    m : positive integer
        truncation weight of the summation
    a : numeric or complex vector
        vector of the "upper" parameters, possibly empty (or `None`)
    b : numeric or complex vector
        vector of the "lower" parameters, possibly empty (or `None`)
    x : numeric or complex vector
        arguments (the eigenvalues of the matrix)
    alpha : positive number, optional
        the alpha parameter; the default is 2

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if a is None or len(a) == 0:
        a = np.array([])
    else:
        a = np.asarray(a)
    if b is None or len(b) == 0:
        b = np.array([])
    else:
        b = np.asarray(b)
    x = np.asarray(x)
    n = len(x)
    if all(x == x[0]):
        return __hypergeomI(m, alpha, a, b, n, x[0])
    def jack(k, beta, c, t, mu, jarray, kappa, nkappa):
        lmu = len(mu)
        for i in range(max(1, k), (np.count_nonzero(mu)+1)):
            u = mu[i-1]
            if lmu == i or u > mu[i]:
                gamma = beta * __betaratio(kappa, mu, i, alpha)
                mup = mu.copy()
                mup[i-1] = u - 1
                mup = mup[mup > 0]
                if len(mup) >= i and u > 1:
                    jack(i, gamma, c + 1, t, mup, jarray, kappa, nkappa)
                else:
                    if nkappa > 1:
                        if len(mup) > 0:
                            jarray[nkappa-1, t-1] += (
                                gamma * jarray[_N(dico, mup)-2, t-2] 
                                * x[t-1]**(c+1)
                            )
                        else:
                            jarray[nkappa-1, t-1] += gamma * x[t-1]**(c+1)
        if k == 0:
            if nkappa > 1:
                jarray[nkappa-1, t-1] += jarray[nkappa-1, t-2]
        else:
            jarray[nkappa-1, t-1] += (
                beta * x[t-1]**c * jarray[_N(dico, mu)-2, t-2]
            )
    def summation(i, z, j, kappa, jarray):
        def go(kappai, zp, s):
            if (
                    i == n or i == 0 and kappai > j 
                    or i > 0 and kappai > min(kappa[-1], j)
                ):
                return s
            kappap = np.concatenate((kappa, [kappai]))
            nkappa = _N(dico, kappap) - 1
            zpp = zp * _T(alpha, a, b, kappap)
            if nkappa > 1 and (len(kappap) == 1 or kappap[1] == 0):
                 jarray[nkappa-1, 0] = (
                     x[0] * (1 + alpha * (kappap[0] - 1)) * jarray[nkappa-2, 0]
                 )
            for t in range(2, n+1):
                jack(0, 1.0, 0, t, kappap, jarray, kappap, nkappa)
            sp = s + zpp * jarray[nkappa-1, n-1]
            if j > kappai and i <= n:
                spp = summation(i+1, zpp, j-kappai, kappap, jarray)
                return go(kappai+1, zpp, sp + spp)
            return go(kappai+1, zpp, sp)
        return go(1, z, 0)
    (dico, Pmn) = __DictParts(m, n)
    T = type(x[0])
    J = np.zeros((Pmn, n), dtype=T)
    J[0, :] = np.cumsum(x)
    return 1 + summation(0, T(1), m, np.empty(0, dtype=int), J)
