# pyhypergeomatrix

<!-- badges: start -->
[![Documentation status](https://readthedocs.org/projects/pyhypergeomatrix/badge/)](http://pyhypergeomatrix.readthedocs.io)
<!-- badges: end -->

*Hypergeometric functions of a matrix argument.*

___

## Evaluation of the hypergeometric function of a matrix argument (Koev & Edelman's algorithm)

Let $(a\_1, \ldots, a\_p)$ and $(b\_1, \ldots, b\_q)$ be two vectors of real or 
complex numbers, possibly empty, $\alpha > 0$ and $X$ a real symmetric or a 
complex Hermitian matrix. 
The corresponding *hypergeometric function of a matrix argument* is defined by 

$${}\_pF\_q^{(\alpha)} \left(\begin{matrix} a\_1, \ldots, a\_p \\\\ b\_1, \ldots, b\_q\end{matrix}; X\right) = \sum\_{k=0}^{\infty}\sum\_{\kappa \vdash k} \frac{{(a\_1)}\_{\kappa}^{(\alpha)} \cdots {(a\_p)}\_{\kappa}^{(\alpha)}} {{(b\_1)}\_{\kappa}^{(\alpha)} \cdots {(b\_q)}\_{\kappa}^{(\alpha)}} \frac{C\_{\kappa}^{(\alpha)}(X)}{k!}.$$

The inner sum is over the integer partitions $\kappa$ of $k$ (which we also 
denote by $|\kappa| = k$). The symbol ${(\cdot)}\_{\kappa}^{(\alpha)}$ is the 
*generalized Pochhammer symbol*, defined by

$${(c)}^{(\alpha)}\_{\kappa} = \prod\_{i=1}^{\ell}\prod\_{j=1}^{\kappa\_i} \left(c - \frac{i-1}{\alpha} + j-1\right)$$

when $\kappa = (\kappa\_1, \ldots, \kappa\_\ell)$. 
Finally, $C\_{\kappa}^{(\alpha)}$ is a *Jack function*. 
Given an integer partition $\kappa$ and $\alpha > 0$, and a 
real symmetric or complex Hermitian matrix $X$ of order $n$, 
the Jack function 

$$C\_{\kappa}^{(\alpha)}(X) = C\_{\kappa}^{(\alpha)}(x\_1, \ldots, x\_n)$$

is a symmetric homogeneous polynomial of degree $|\kappa|$ in the 
eigen values $x\_1$, $\ldots$, $x\_n$ of $X$. 

The series defining the hypergeometric function does not always converge. 
See the references for a discussion about the convergence. 

The inner sum in the definition of the hypergeometric function is over 
all partitions $\kappa \vdash k$ but actually 
$C\_{\kappa}^{(\alpha)}(X) = 0$ when $\ell(\kappa)$, the number of non-zero 
entries of $\kappa$, is strictly greater than $n$.

For $\alpha=1$, $C\_{\kappa}^{(\alpha)}$ is a *Schur polynomial* and it is 
a *zonal polynomial* for $\alpha = 2$. 
In random matrix theory, the hypergeometric function appears for $\alpha=2$ 
and $\alpha$ is omitted from the notation, implicitely assumed to be $2$. 

Koev and Edelman (2006) provided an efficient algorithm for the evaluation 
of the truncated series 

$$\sideset{\_p^m}{\_q^{(\alpha)}}F \left(\begin{matrix} a\_1, \ldots, a\_p \\\\ b\_1, \ldots, b\_q\end{matrix}; X\right) = \sum\_{k=0}^{m}\sum\_{\kappa \vdash k} \frac{{(a\_1)}\_{\kappa}^{(\alpha)} \cdots {(a\_p)}\_{\kappa}^{(\alpha)}} {{(b\_1)}\_{\kappa}^{(\alpha)} \cdots {(b\_q)}\_{\kappa}^{(\alpha)}} 
\frac{C\_{\kappa}^{(\alpha)}(X)}{k!}.$$

Hereafter, $m$ is called the *truncation weight of the summation* 
(because $|\kappa|$ is called the weight of $\kappa$), the vector 
$(a\_1, \ldots, a\_p)$ is called the vector of *upper parameters* while 
the vector $(b\_1, \ldots, b\_q)$ is called the vector of *lower parameters*. 
The user has to supply the vector $(x\_1, \ldots, x\_n)$ of the eigenvalues 
of $X$. 

For example, to compute

$$\sideset{\_2^{15}}{\_3^{(2)}}F \left(\begin{matrix} 3, 4 \\\\ 5, 6, 7\end{matrix}; 0.1, 0.4\right)$$

you have to enter 

```haskell
>>> from pyhypergeomatrix.hypergeomat import hypergeomPQ
>>> hypergeomPQ(15, [3, 4], [5, 6, 7], [0.1, 0.4], 2)
```

We said that the hypergeometric function is defined for a real symmetric 
matrix or a complex Hermitian matrix $X$. Thus the eigenvalues of $X$ 
are real. However we do not impose this restriction in `pyhypergeomatrix`. 
The user can enter any list of real or complex numbers for the eigenvalues. 


### Univariate case

For $n = 1$, the hypergeometric function of a matrix argument is known as the 
[generalized hypergeometric function](https://mathworld.wolfram.com/HypergeometricFunction.html). 
It does not depend on $\alpha$. The case of $\sideset{\_{2\thinspace}^{}}{\_1^{}}F$ is the most known, 
this is the Gauss hypergeometric function. Let's check a value. It is known that

$$\sideset{\_{2\thinspace}^{}}{\_1^{}}F \left(\begin{matrix} 1/4, 1/2 \\\\ 3/4\end{matrix}; 80/81\right) = 1.8.$$

Since $80/81$ is close to $1$, the convergence is slow. We compute the truncated series below 
for $m = 300$.

```python
>>> from pyhypergeomatrix.hypergeomat import hypergeomPQ
>>> hypergeomPQ(300, [1/4, 1/2], [3/4], [80/81])
1.79900265281923
```


## References

- Plamen Koev and Alan Edelman. 
*The efficient evaluation of the hypergeometric function of a matrix argument*.
Mathematics of computation, vol. 75, n. 254, 833-846, 2006.

- Robb Muirhead. 
*Aspects of multivariate statistical theory*. 
Wiley series in probability and mathematical statistics. 
Probability and mathematical statistics. 
John Wiley & Sons, New York, 1982.

- A. K. Gupta and D. K. Nagar. 
*Matrix variate distributions*. 
Chapman and Hall, 1999.