#!/usr/bin/env python3
"""
Subspace Analysis: exhaustive search for the Oil subspace in a toy MAYO instance over GF(16).

Attempts to recover the secret Oil subspace from the public key alone by:
  1. Enumerating all vectors in GF(16)^n that evaluate to zero under every
     public quadratic form (the "roots").
  2. Applying an addition test: a true Oil vector v must satisfy
     v + w in roots for every other Oil vector w.  Over GF(16) the Oil
     subspace has 16^o - 1 non-zero vectors, so expected_matches = 16^o - 2.
  3. Comparing the recovered subspace to the actual secret Oil space.

Run with:  sage -python old_analysis/subspace_analysis.py
Requires:  SageMath, MAYO-sage
Output:    console

Note: brute-force is feasible only for tiny n.  With n=4 the search space is
      16^4 = 65,536 vectors.  Increase n with care (n=5 → ~1 M, n=6 → ~16 M).
"""

import sys
import os
from itertools import product
from sage.all import block_matrix, matrix, vector

sys.path.append(os.path.abspath('MAYO-sage'))

try:
    from sagelib.mayo import Mayo, F16, R, z
    from sagelib.utilities import decode_matrix, decode_matrices
except ImportError as e:
    sys.exit("Error loading MAYO-sage. " + str(e))

def main():
    # n=4 keeps the search space at 16^4 = 65,536 vectors.
    # o=2  => Oil subspace has 16^2 - 1 = 255 non-zero vectors over GF(16).
    toy_params = {
        "name": "mayo_toy",
        "q": 16, "m": 4, "n": 4, "o": 2, "k": 2,
        "sk_salt_bytes": 16, "pk_bytes": 16, "digest_bytes": 16,
        "f": R.irreducible_element(4, algorithm="random")
    }
    mayo = Mayo(toy_params)

    csk, cpk = mayo.compact_key_gen()
    esk = mayo.expand_sk(csk)
    epk = mayo.expand_pk(cpk)

    # Recover the actual Oil space from the secret key.
    O_bytes = esk[mayo.sk_seed_bytes : mayo.sk_seed_bytes + mayo.O_bytes]
    O = decode_matrix(O_bytes, mayo.n - mayo.o, mayo.o)
    # Basis of Oil space: columns of [O^T | I_o]^T
    oil_basis_matrix = block_matrix([[O], [matrix.identity(F16, mayo.o)]])
    oil_space = oil_basis_matrix.column_space()
    print(f"Actual Oil Space Dimension: {oil_space.dimension()}")
    print(f"Non-zero Oil vectors: {16**mayo.o - 1}")

    # Build the full public-key matrices.
    P1 = decode_matrices(epk[:mayo.P1_bytes], mayo.m, mayo.n - mayo.o, mayo.n - mayo.o, triangular=True)
    P2 = decode_matrices(epk[mayo.P1_bytes : mayo.P1_bytes + mayo.P2_bytes], mayo.m, mayo.n - mayo.o, mayo.o, triangular=False)
    P3 = decode_matrices(epk[mayo.P1_bytes + mayo.P2_bytes : mayo.P1_bytes + mayo.P2_bytes + mayo.P3_bytes], mayo.m, mayo.o, mayo.o, triangular=True)
    P = [block_matrix([[P1[a], P2[a]], [matrix(F16, mayo.o, mayo.n - mayo.o), P3[a]]]) for a in range(mayo.m)]

    # Step 1: Enumerate all roots of the public quadratic system.
    total_vectors = 16**mayo.n
    print(f"\nSearch space: {total_vectors} vectors")

    roots = []
    for coeffs in product(F16, repeat=mayo.n):
        x = vector(F16, coeffs)
        if x.is_zero():
            continue
        if all((x * P[a]) * x == 0 for a in range(mayo.m)):
            roots.append(x)

    print(f"Found {len(roots)} roots.")

    # Step 2: Addition test.
    # A GF(16)-subspace of dimension o contains 16^o - 1 non-zero vectors, so
    # each Oil vector v must pair with at least 16^o - 2 other roots w such
    # that v + w is also a root.
    print("\nRunning the Addition Test...")
    roots_set = set(tuple(r) for r in roots)
    expected_matches = 16**mayo.o - 2

    valid_oil_vectors = []
    for r in roots:
        matches = sum(
            1 for r2 in roots
            if r2 != r and tuple(r + r2) in roots_set
        )
        if matches >= expected_matches:
            valid_oil_vectors.append(r)

    print(f"Vectors passing addition test: {len(valid_oil_vectors)}  (expected {16**mayo.o - 1})")

    # Step 3: Compare recovered subspace to the secret Oil space.
    if len(valid_oil_vectors) > 0:
        recovered_matrix = matrix(F16, valid_oil_vectors).transpose()  # vectors as columns
        recovered_space = recovered_matrix.column_space()
        print(f"Recovered Subspace Dimension: {recovered_space.dimension()}")

        if recovered_space == oil_space:
            print("SUCCESS: The recovered subspace perfectly matches the Secret Oil Space!")
        else:
            intersection_dim = recovered_space.intersection(oil_space).dimension()
            print(f"FAILURE: Space does not match. (Intersection dimension: {intersection_dim})")
    else:
        print("FAILURE: Could not find any valid Oil vectors.")

if __name__ == '__main__':
    main()
