from functools import wraps
from itertools import combinations_with_replacement as cwr
from itertools import permutations, product

import numba
import numpy as np


# from fractions import Fraction
@numba.jit
def Fraction(p, q):
    return p / q


@numba.jit(nogil=True)
def D(counts):
    n1 = counts[0]
    n2 = counts[1]
    n3 = counts[2]
    n4 = counts[3]
    n5 = counts[4]
    n6 = counts[5]
    n7 = counts[6]
    n8 = counts[7]
    n9 = counts[8]
    nd = np.sum(counts)
    numer = (
        -Fraction(n2 * n4, 4)
        - Fraction(n3 * n4, 2)
        + Fraction(n1 * n5, 4)
        - Fraction(n3 * n5, 4)
        + Fraction(n1 * n6, 2)
        + Fraction(n2 * n6, 4)
        - Fraction(n2 * n7, 2)
        - Fraction(n3 * n7, 1)
        - Fraction(n5 * n7, 4)
        - Fraction(n6 * n7, 2)
        + Fraction(n1 * n8, 2)
        - Fraction(n3 * n8, 2)
        + Fraction(n4 * n8, 4)
        - Fraction(n6 * n8, 4)
        + Fraction(n1 * n9, 1)
        + Fraction(n2 * n9, 2)
        + Fraction(n4 * n9, 2)
        + Fraction(n5 * n9, 4)
    )
    denom = nd * (nd - 1)
    return 2.0 * (numer / denom)


Dhat = D


@numba.jit(nogil=True)
def D2(counts):
    n1 = counts[0]
    n2 = counts[1]
    n3 = counts[2]
    n4 = counts[3]
    n5 = counts[4]
    n6 = counts[5]
    n7 = counts[6]
    n8 = counts[7]
    n9 = counts[8]
    n = np.sum(counts)
    numer = (
        Fraction(
            n2 * n4
            - n2**2 * n4
            + 4 * n3 * n4
            - 4 * n2 * n3 * n4
            - 4 * n3**2 * n4
            - n2 * n4**2
            - 4 * n3 * n4**2
            + n1 * n5
            - n1**2 * n5
            + n3 * n5
            + 2 * n1 * n3 * n5
            - n3**2 * n5
            - 4 * n3 * n4 * n5
            - n1 * n5**2
            - n3 * n5**2
            + 4 * n1 * n6
            - 4 * n1**2 * n6
            + n2 * n6
            - 4 * n1 * n2 * n6
            - n2**2 * n6
            + 2 * n2 * n4 * n6
            - 4 * n1 * n5 * n6
            - 4 * n1 * n6**2
            - n2 * n6**2
            + 4 * n2 * n7
            - 4 * n2**2 * n7
            + 16 * n3 * n7
            - 16 * n2 * n3 * n7
            - 16 * n3**2 * n7
            - 4 * n2 * n4 * n7
            - 16 * n3 * n4 * n7
            + n5 * n7
            + 2 * n1 * n5 * n7
            - 4 * n2 * n5 * n7
            - 18 * n3 * n5 * n7
            - n5**2 * n7
            + 4 * n6 * n7
            + 8 * n1 * n6 * n7
            - 16 * n3 * n6 * n7
            - 4 * n5 * n6 * n7
            - 4 * n6**2 * n7
            - 4 * n2 * n7**2
            - 16 * n3 * n7**2
            - n5 * n7**2
            - 4 * n6 * n7**2
            + 4 * n1 * n8
            - 4 * n1**2 * n8
            + 4 * n3 * n8
            + 8 * n1 * n3 * n8
            - 4 * n3**2 * n8
            + n4 * n8
            - 4 * n1 * n4 * n8
            + 2 * n2 * n4 * n8
            - n4**2 * n8
            - 4 * n1 * n5 * n8
            - 4 * n3 * n5 * n8
            + n6 * n8
            + 2 * n2 * n6 * n8
            - 4 * n3 * n6 * n8
            + 2 * n4 * n6 * n8
            - n6**2 * n8
            - 16 * n3 * n7 * n8
            - 4 * n6 * n7 * n8
            - 4 * n1 * n8**2
            - 4 * n3 * n8**2
            - n4 * n8**2
            - n6 * n8**2
            + 16 * n1 * n9
            - 16 * n1**2 * n9
            + 4 * n2 * n9
            - 16 * n1 * n2 * n9
            - 4 * n2**2 * n9
            + 4 * n4 * n9
            - 16 * n1 * n4 * n9
            + 8 * n3 * n4 * n9
            - 4 * n4**2 * n9
            + n5 * n9
            - 18 * n1 * n5 * n9
            - 4 * n2 * n5 * n9
            + 2 * n3 * n5 * n9
            - 4 * n4 * n5 * n9
            - n5**2 * n9
            - 16 * n1 * n6 * n9
            - 4 * n2 * n6 * n9
            + 8 * n2 * n7 * n9
            + 2 * n5 * n7 * n9
            - 16 * n1 * n8 * n9
            - 4 * n4 * n8 * n9
            - 16 * n1 * n9**2
            - 4 * n2 * n9**2
            - 4 * n4 * n9**2
            - n5 * n9**2,
            16,
        )
        + (
            -(
                (Fraction(n2, 2) + n3 + Fraction(n5, 4) + Fraction(n6, 2))
                * (Fraction(n4, 2) + Fraction(n5, 4) + n7 + Fraction(n8, 2))
            )
            + (n1 + Fraction(n2, 2) + Fraction(n4, 2) + Fraction(n5, 4))
            * (Fraction(n5, 4) + Fraction(n6, 2) + Fraction(n8, 2) + n9)
        )
        ** 2
    )
    denom = n * (n - 1) * (n - 2) * (n - 3)
    return 4.0 * (numer / denom)


@numba.jit(nogil=True)
def Dz(counts):
    n1 = counts[0]
    n2 = counts[1]
    n3 = counts[2]
    n4 = counts[3]
    n5 = counts[4]
    n6 = counts[5]
    n7 = counts[6]
    n8 = counts[7]
    n9 = counts[8]
    n = np.sum(counts)
    numer = Fraction(
        -(n2 * n4)
        + 3 * n1 * n2 * n4
        + n2**2 * n4
        + 2 * n3 * n4
        + 4 * n1 * n3 * n4
        - n2 * n3 * n4
        - 4 * n3**2 * n4
        + n2 * n4**2
        + 2 * n3 * n4**2
        + 2 * n1 * n5
        - 3 * n1**2 * n5
        - n1 * n2 * n5
        + 2 * n3 * n5
        + 2 * n1 * n3 * n5
        - n2 * n3 * n5
        - 3 * n3**2 * n5
        - n1 * n4 * n5
        + n3 * n4 * n5
        + 2 * n1 * n6
        - 4 * n1**2 * n6
        - n2 * n6
        - n1 * n2 * n6
        + n2**2 * n6
        + 4 * n1 * n3 * n6
        + 3 * n2 * n3 * n6
        - 2 * n1 * n4 * n6
        - 2 * n2 * n4 * n6
        - 2 * n3 * n4 * n6
        + n1 * n5 * n6
        - n3 * n5 * n6
        + 2 * n1 * n6**2
        + n2 * n6**2
        + 2 * n2 * n7
        + 4 * n1 * n2 * n7
        + 2 * n2**2 * n7
        + 8 * n3 * n7
        + 4 * n1 * n3 * n7
        - 4 * n3**2 * n7
        - n2 * n4 * n7
        + 2 * n5 * n7
        + 2 * n1 * n5 * n7
        + n2 * n5 * n7
        + 2 * n3 * n5 * n7
        - n4 * n5 * n7
        + 2 * n6 * n7
        - n2 * n6 * n7
        - 2 * n4 * n6 * n7
        + n5 * n6 * n7
        + 2 * n6**2 * n7
        - 4 * n2 * n7**2
        - 4 * n3 * n7**2
        - 3 * n5 * n7**2
        - 4 * n6 * n7**2
        + 2 * n1 * n8
        - 4 * n1**2 * n8
        - 2 * n1 * n2 * n8
        + 2 * n3 * n8
        - 2 * n2 * n3 * n8
        - 4 * n3**2 * n8
        - n4 * n8
        - n1 * n4 * n8
        - 2 * n2 * n4 * n8
        - n3 * n4 * n8
        + n4**2 * n8
        + n1 * n5 * n8
        + n3 * n5 * n8
        - n6 * n8
        - n1 * n6 * n8
        - 2 * n2 * n6 * n8
        - n3 * n6 * n8
        - 2 * n4 * n6 * n8
        + n6**2 * n8
        + 4 * n1 * n7 * n8
        - 2 * n2 * n7 * n8
        + 3 * n4 * n7 * n8
        - n5 * n7 * n8
        - n6 * n7 * n8
        + 2 * n1 * n8**2
        + 2 * n3 * n8**2
        + n4 * n8**2
        + n6 * n8**2
        + 8 * n1 * n9
        - 4 * n1**2 * n9
        + 2 * n2 * n9
        + 2 * n2**2 * n9
        + 4 * n1 * n3 * n9
        + 4 * n2 * n3 * n9
        + 2 * n4 * n9
        - n2 * n4 * n9
        + 2 * n4**2 * n9
        + 2 * n5 * n9
        + 2 * n1 * n5 * n9
        + n2 * n5 * n9
        + 2 * n3 * n5 * n9
        + n4 * n5 * n9
        - n2 * n6 * n9
        - 2 * n4 * n6 * n9
        - n5 * n6 * n9
        + 4 * n1 * n7 * n9
        + 4 * n3 * n7 * n9
        + 4 * n4 * n7 * n9
        + 2 * n5 * n7 * n9
        + 4 * n6 * n7 * n9
        - 2 * n2 * n8 * n9
        + 4 * n3 * n8 * n9
        - n4 * n8 * n9
        - n5 * n8 * n9
        + 3 * n6 * n8 * n9
        - 4 * n1 * n9**2
        - 4 * n2 * n9**2
        - 4 * n4 * n9**2
        - 3 * n5 * n9**2,
        4,
    ) + (-n1 + n3 - n4 + n6 - n7 + n9) * (-n1 - n2 - n3 + n7 + n8 + n9) * (
        -(
            (Fraction(n2, 2) + n3 + Fraction(n5, 4) + Fraction(n6, 2))
            * (Fraction(n4, 2) + Fraction(n5, 4) + n7 + Fraction(n8, 2))
        )
        + (n1 + Fraction(n2, 2) + Fraction(n4, 2) + Fraction(n5, 4))
        * (Fraction(n5, 4) + Fraction(n6, 2) + Fraction(n8, 2) + n9)
    )
    denom = n * (n - 1) * (n - 2) * (n - 3)
    return 2.0 * (numer / denom)


@numba.jit(nogil=True)
def pi2(counts):
    n1 = counts[0]
    n2 = counts[1]
    n3 = counts[2]
    n4 = counts[3]
    n5 = counts[4]
    n6 = counts[5]
    n7 = counts[6]
    n8 = counts[7]
    n9 = counts[8]
    n = np.sum(counts)
    numer = (n1 + n2 + n3 + n4 / 2.0 + n5 / 2.0 + n6 / 2.0) * (
        n1 + n2 / 2.0 + n4 + n5 / 2.0 + n7 + n8 / 2.0
    ) * (n2 / 2.0 + n3 + n5 / 2.0 + n6 + n8 / 2.0 + n9) * (
        n4 / 2.0 + n5 / 2.0 + n6 / 2.0 + n7 + n8 + n9
    ) + (
        13 * n2 * n4
        - 16 * n1 * n2 * n4
        - 11 * n2**2 * n4
        + 16 * n3 * n4
        - 28 * n1 * n3 * n4
        - 24 * n2 * n3 * n4
        - 8 * n3**2 * n4
        - 11 * n2 * n4**2
        - 20 * n3 * n4**2
        - 6 * n5
        + 12 * n1 * n5
        - 4 * n1**2 * n5
        + 17 * n2 * n5
        - 20 * n1 * n2 * n5
        - 11 * n2**2 * n5
        + 12 * n3 * n5
        - 28 * n1 * n3 * n5
        - 20 * n2 * n3 * n5
        - 4 * n3**2 * n5
        + 17 * n4 * n5
        - 20 * n1 * n4 * n5
        - 32 * n2 * n4 * n5
        - 40 * n3 * n4 * n5
        - 11 * n4**2 * n5
        + 11 * n5**2
        - 16 * n1 * n5**2
        - 17 * n2 * n5**2
        - 16 * n3 * n5**2
        - 17 * n4 * n5**2
        - 6 * n5**3
        + 16 * n1 * n6
        - 8 * n1**2 * n6
        + 13 * n2 * n6
        - 24 * n1 * n2 * n6
        - 11 * n2**2 * n6
        - 28 * n1 * n3 * n6
        - 16 * n2 * n3 * n6
        + 24 * n4 * n6
        - 36 * n1 * n4 * n6
        - 38 * n2 * n4 * n6
        - 36 * n3 * n4 * n6
        - 20 * n4**2 * n6
        + 17 * n5 * n6
        - 40 * n1 * n5 * n6
        - 32 * n2 * n5 * n6
        - 20 * n3 * n5 * n6
        - 42 * n4 * n5 * n6
        - 17 * n5**2 * n6
        - 20 * n1 * n6**2
        - 11 * n2 * n6**2
        - 20 * n4 * n6**2
        - 11 * n5 * n6**2
        + 16 * n2 * n7
        - 28 * n1 * n2 * n7
        - 20 * n2**2 * n7
        + 16 * n3 * n7
        - 48 * n1 * n3 * n7
        - 44 * n2 * n3 * n7
        - 16 * n3**2 * n7
        - 24 * n2 * n4 * n7
        - 44 * n3 * n4 * n7
        + 12 * n5 * n7
        - 28 * n1 * n5 * n7
        - 40 * n2 * n5 * n7
        - 48 * n3 * n5 * n7
        - 20 * n4 * n5 * n7
        - 16 * n5**2 * n7
        + 16 * n6 * n7
        - 48 * n1 * n6 * n7
        - 48 * n2 * n6 * n7
        - 44 * n3 * n6 * n7
        - 36 * n4 * n6 * n7
        - 40 * n5 * n6 * n7
        - 20 * n6**2 * n7
        - 8 * n2 * n7**2
        - 16 * n3 * n7**2
        - 4 * n5 * n7**2
        - 8 * n6 * n7**2
        + 16 * n1 * n8
        - 8 * n1**2 * n8
        + 24 * n2 * n8
        - 36 * n1 * n2 * n8
        - 20 * n2**2 * n8
        + 16 * n3 * n8
        - 48 * n1 * n3 * n8
        - 36 * n2 * n3 * n8
        - 8 * n3**2 * n8
        + 13 * n4 * n8
        - 24 * n1 * n4 * n8
        - 38 * n2 * n4 * n8
        - 48 * n3 * n4 * n8
        - 11 * n4**2 * n8
        + 17 * n5 * n8
        - 40 * n1 * n5 * n8
        - 42 * n2 * n5 * n8
        - 40 * n3 * n5 * n8
        - 32 * n4 * n5 * n8
        - 17 * n5**2 * n8
        + 13 * n6 * n8
        - 48 * n1 * n6 * n8
        - 38 * n2 * n6 * n8
        - 24 * n3 * n6 * n8
        - 38 * n4 * n6 * n8
        - 32 * n5 * n6 * n8
        - 11 * n6**2 * n8
        - 28 * n1 * n7 * n8
        - 36 * n2 * n7 * n8
        - 44 * n3 * n7 * n8
        - 16 * n4 * n7 * n8
        - 20 * n5 * n7 * n8
        - 24 * n6 * n7 * n8
        - 20 * n1 * n8**2
        - 20 * n2 * n8**2
        - 20 * n3 * n8**2
        - 11 * n4 * n8**2
        - 11 * n5 * n8**2
        - 11 * n6 * n8**2
        + 16 * n1 * n9
        - 16 * n1**2 * n9
        + 16 * n2 * n9
        - 44 * n1 * n2 * n9
        - 20 * n2**2 * n9
        - 48 * n1 * n3 * n9
        - 28 * n2 * n3 * n9
        + 16 * n4 * n9
        - 44 * n1 * n4 * n9
        - 48 * n2 * n4 * n9
        - 48 * n3 * n4 * n9
        - 20 * n4**2 * n9
        + 12 * n5 * n9
        - 48 * n1 * n5 * n9
        - 40 * n2 * n5 * n9
        - 28 * n3 * n5 * n9
        - 40 * n4 * n5 * n9
        - 16 * n5**2 * n9
        - 44 * n1 * n6 * n9
        - 24 * n2 * n6 * n9
        - 36 * n4 * n6 * n9
        - 20 * n5 * n6 * n9
        - 48 * n1 * n7 * n9
        - 48 * n2 * n7 * n9
        - 48 * n3 * n7 * n9
        - 28 * n4 * n7 * n9
        - 28 * n5 * n7 * n9
        - 28 * n6 * n7 * n9
        - 44 * n1 * n8 * n9
        - 36 * n2 * n8 * n9
        - 28 * n3 * n8 * n9
        - 24 * n4 * n8 * n9
        - 20 * n5 * n8 * n9
        - 16 * n6 * n8 * n9
        - 16 * n1 * n9**2
        - 8 * n2 * n9**2
        - 8 * n4 * n9**2
        - 4 * n5 * n9**2
    ) / 16.0
    denom = n * (n - 1) * (n - 2) * (n - 3)
    return numer / denom
