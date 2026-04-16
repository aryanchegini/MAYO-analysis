#!/usr/bin/env python3
"""
GF(16) field setup and MQ instance generation for the MAYO analysis experiments.

Contains all mathematical primitives: field arithmetic, Groebner-basis
polynomial helpers, SAT encoding tables, whipped-map construction, and the
planted-solution variable reduction.

Run directly to execute the built-in correctness tests.
"""

from sage.all import GF, PolynomialRing, matrix, vector, set_random_seed

# GF(16) field setup. MAYO uses GF(16) with irreducible polynomial x^4 + x + 1.
F16 = GF(16, 'a')
assert F16.gen()**4 + F16.gen() + 1 == 0, "Expected GF(16) with poly a^4+a+1"
INT_TO_F16 = [F16.from_integer(i) for i in range(16)]
F16_TO_INT = {v: k for k, v in enumerate(INT_TO_F16)}


# GF(16) arithmetic tables (used by the SAT encoding in H1)
def _build_gf16_mul_table():
    """
    Precompute the 16x16 multiplication table for GF(16) = GF(2)[x]/(x^4+x+1).
    Carry-less multiply with reduction: if high bit set, XOR with 0b0011 = x+1.
    """
    table = [[0] * 16 for _ in range(16)]
    for a in range(16):
        for b in range(16):
            r, aa, bb = 0, a, b
            for _ in range(4):
                if bb & 1:
                    r ^= aa
                bb >>= 1
                carry = aa & 8
                aa = (aa << 1) & 0xF
                if carry:
                    aa ^= 3
            table[a][b] = r
    return table


_GF16_MUL = _build_gf16_mul_table()

# For each constant c, the 4x4 GF(2) matrix representing the linear map x -> c*x.
# _GF16_CONST_MAT[c][output_bit] is a 4-bit mask: bit j set means output_bit
# includes a contribution from input_bit j.
_GF16_CONST_MAT = {}
for _c in range(16):
    _cols = [_GF16_MUL[_c][1 << _j] for _j in range(4)]
    _GF16_CONST_MAT[_c] = [
        sum(int((_cols[_j] >> _i) & 1) << _j for _j in range(4))
        for _i in range(4)
    ]

# Boolean decomposition of GF(16) multiplication x*y into AND-products of bit pairs.
# Derived by carry-less multiply of 4-bit values mod x^4+x+1:
#   r0 = x0y0 ^ x1y3 ^ x2y2 ^ x3y1
#   r1 = x0y1 ^ x1y0 ^ x1y3 ^ x2y2 ^ x3y1 ^ x2y3 ^ x3y2
#   r2 = x0y2 ^ x1y1 ^ x2y0 ^ x2y3 ^ x3y2 ^ x3y3
#   r3 = x0y3 ^ x1y2 ^ x2y1 ^ x3y0 ^ x3y3
_GF16_MUL_PATTERN = [
    [(0,0),(1,3),(2,2),(3,1)],
    [(0,1),(1,0),(1,3),(2,2),(3,1),(2,3),(3,2)],
    [(0,2),(1,1),(2,0),(2,3),(3,2),(3,3)],
    [(0,3),(1,2),(2,1),(3,0),(3,3)],
]


# H1 instance generation
def generate_mq_coeffs(n, m, seed):
    """
    Generate a random dense MQ instance over GF(16).

    Coefficients are sampled uniformly from GF(16) including zero, giving
    an expected density of 15/16 non-zero terms per monomial.

    The target is computed as F(x0) for a randomly sampled x0 in GF(16)^n,
    so the system is guaranteed to have at least one solution.  This matches
    Bard's methodology: the polynomials are fully random and the target is
    derived by evaluation rather than sampled independently.  An independently
    sampled random target may not be in the image of F, producing UNSAT
    instances that cause SAT solvers to spend time on infeasibility proofs
    rather than solution finding.

    Returns
    -------
    polys  : list of m dicts {(i,j): int}, i<=j, upper-triangular form
    target : list of m ints in 0..15
    """
    set_random_seed(seed)
    polys = []
    for _ in range(m):
        d = {}
        for i in range(n):
            for j in range(i, n):
                d[(i, j)] = F16_TO_INT[F16.random_element()]
        polys.append(d)

    # Plant a solution: sample x0, compute target = F(x0).
    x0 = [F16_TO_INT[F16.random_element()] for _ in range(n)]
    target = []
    for a in range(m):
        val = 0
        for (i, j), c in polys[a].items():
            if c != 0:
                val ^= _GF16_MUL[c][_GF16_MUL[x0[i]][x0[j]]]
        target.append(val)

    return polys, target


def coeffs_to_sage_polys(n, m, polys_data, target_data):
    """Convert coefficient dicts to SageMath polynomial objects over GF(16)."""
    R  = PolynomialRing(F16, n, 'x')
    xs = R.gens()
    polys = []
    for d in polys_data:
        p = R.zero()
        for (i, j), c in d.items():
            coeff = INT_TO_F16[c]
            if coeff != F16.zero():
                p += coeff * xs[i] * xs[j]
        polys.append(p)
    return R, xs, polys, [INT_TO_F16[c] for c in target_data]


def validate_solution(m, polys_data, target_data, solution_vals):
    """Return True iff solution_vals satisfies all m equations."""
    for a in range(m):
        val = 0
        for (i, j), c in polys_data[a].items():
            if c != 0:
                val ^= _GF16_MUL[c][_GF16_MUL[solution_vals[i]][solution_vals[j]]]
        if val != target_data[a]:
            return False
    return True


# H2 instance generation
def companion_matrix(f_poly, m):
    """
    Build the m x m companion matrix of f_poly over F16.

    f_poly is a monic polynomial of degree m in F16[z].  Convention: C acts
    on column vectors so that Cv gives multiplication by z in F16[z]/(f_poly)
    when the vector encodes polynomial coefficients.
    """
    C = matrix(F16, m, m)
    for i in range(m - 1):
        C[i + 1, i] = F16.one()
    for i in range(m):
        # f_poly = z^m + ... + c_i*z^i + ...  =>  z^m = -sum(c_i z^i)
        # In char 2, -c_i = c_i.
        C[i, m - 1] = -f_poly[i]
    return C


def generate_whipped_instance(n, m, o, k, seed):
    """
    Build a genuine UOV-structured MAYO whipped map P*.

    Parameters
    ----------
    n : int  -- total variables per block (v = n-o vinegar, o oil)
    m : int  -- number of output components
    o : int  -- oil dimension; must satisfy o < n and v = n-o > o
    k : int  -- number of signature blocks
    seed     -- SageMath random seed

    Structure
    ---------
    Each of the m underlying n x n matrices P^(a) has the form:

        [ P1  P2 ]
        [  0   0 ]

    where P1 is a (v x v) upper-triangular matrix and P2 is a (v x o) matrix.
    The oil-oil block (P3) is identically zero, which is the UOV condition: the
    quadratic form P^(a)(x, x) vanishes whenever x is a pure oil vector
    (vinegar components = 0).  Using symmetric matrices for P^(a) would cause
    cross terms in the whipped sum to cancel in characteristic 2, so
    upper-triangular (not symmetric) matrices are used throughout.

    Target
    ------
    A planted solution x0 is sampled uniformly from F16^{kn}.  The target
    t = P*(x0) is computed by polynomial evaluation, so the returned system
    P*(x) - t = 0 is guaranteed to have at least one solution.

    Returns (R, xs, equations, target, kn, x0_int) where x0_int is the
    planted solution as a list of kn integers in 0..15.
    """
    assert o < n and n - o > o, f"Need o < n/2, got n={n}, o={o}"
    set_random_seed(seed)
    v = n - o

    # Build m upper-triangular P matrices with zero oil-oil block.
    P_mats = []
    for _ in range(m):
        M = matrix(F16, n, n)
        for r in range(n):
            for c in range(r, n):
                if r < v: # rows in vinegar block: fill P1 and P2
                    M[r, c] = F16.random_element()
                # rows in oil block (r >= v): P3 = 0, leave as zero
        P_mats.append(M)

    F16_poly_ring = F16['z']
    f_irred       = F16_poly_ring.irreducible_element(m)
    C_mat         = companion_matrix(f_irred, m)

    emulsifier_mats = []
    ell = 0
    for i in range(k):
        for j in range(k - 1, i - 1, -1):
            emulsifier_mats.append((i, j, C_mat ** ell))
            ell += 1

    kn  = k * n
    R   = PolynomialRing(F16, kn, 'x')
    xs  = R.gens()
    s   = [list(xs[i * n:(i + 1) * n]) for i in range(k)]

    y_polys = [R.zero() for _ in range(m)]

    for i, j, E_ell in emulsifier_mats:
        u = [R.zero() for _ in range(m)]
        for a in range(m):
            for r in range(n):
                for c in range(n):
                    coeff = P_mats[a][r, c]
                    if coeff == F16.zero():
                        continue
                    u[a] += coeff * s[i][r] * s[j][c]
            if i != j:
                for r in range(n):
                    for c in range(n):
                        coeff = P_mats[a][r, c]
                        if coeff == F16.zero():
                            continue
                        u[a] += coeff * s[j][r] * s[i][c]

        for b in range(m):
            for a in range(m):
                coeff = E_ell[b, a]
                if coeff != F16.zero():
                    y_polys[b] += coeff * u[a]

    # Plant a solution: sample x0 uniformly, compute t = P*(x0).
    x0_gf   = [F16.random_element() for _ in range(kn)]
    x0_sub  = {xs[i]: x0_gf[i] for i in range(kn)}
    target  = vector(F16, [y_polys[a].subs(x0_sub) for a in range(m)])
    x0_int  = [F16_TO_INT[val] for val in x0_gf]

    equations = [y_polys[a] - target[a] for a in range(m)]
    return R, xs, equations, target, kn, x0_int


def generate_random_instance(kn, m, seed):
    """
    Generate a uniformly random quadratic map Q: F16^{kn} -> F16^m.

    A planted solution y0 is sampled and used to compute the target, so the
    returned system Q(y) - t = 0 is guaranteed to have at least one solution.

    Returns (R, ys, equations, target, y0_int) where y0_int is the planted
    solution as a list of kn integers in 0..15.
    """
    set_random_seed(seed)
    R  = PolynomialRing(F16, kn, 'y')
    ys = R.gens()
    polys = []
    for _ in range(m):
        p = R.zero()
        for i in range(kn):
            for j in range(i, kn):
                c = F16.random_element()
                if c != F16.zero():
                    p += c * ys[i] * ys[j]
        polys.append(p)

    # Plant a solution: sample y0 uniformly, compute t = Q(y0).
    y0_gf   = [F16.random_element() for _ in range(kn)]
    y0_sub  = {ys[i]: y0_gf[i] for i in range(kn)}
    target  = vector(F16, [polys[a].subs(y0_sub) for a in range(m)])
    y0_int  = [F16_TO_INT[val] for val in y0_gf]

    equations = [polys[a] - target[a] for a in range(m)]
    return R, ys, equations, target, y0_int


def reduce_with_planted_solution(R, xs, equations, kn, m, x0_int, seed):
    """
    Fix kn-m variables to their values in x0_int, leaving m free variables.

    Unlike random variable fixing, using the planted solution's own coordinates
    for the fixed variables guarantees the reduced system has at least one
    solution (the free coordinates of x0_int).  This means the Groebner basis
    is always measuring preimage-finding difficulty, not infeasibility
    certification -- which is the correct comparison for H2.

    The choice of which m variables are left free is random (derived from
    seed) but independent of the planted solution values.

    Returns
    -------
    equations_reduced : list of m polynomials in R (only free vars appear)
    field_eqs         : list of m field equations x_i^16 - x_i (free vars)
    free_indices      : list of m variable indices kept as unknowns
    """
    import random as _rng
    rng = _rng.Random(seed ^ 0xDEAD_BEEF)

    all_indices   = list(range(kn))
    rng.shuffle(all_indices)
    free_indices  = sorted(all_indices[:m])
    fixed_indices = sorted(all_indices[m:])

    fixed_vals = {idx: INT_TO_F16[x0_int[idx]] for idx in fixed_indices}
    sub_dict   = {xs[idx]: val for idx, val in fixed_vals.items()}

    equations_reduced = [eq.subs(sub_dict) for eq in equations]
    field_eqs         = [xs[idx]**16 - xs[idx] for idx in free_indices]

    return equations_reduced, field_eqs, free_indices


# Correctness tests
if __name__ == "__main__":
    print("Testing generate_mq_coeffs (planted solution)...")
    polys_t, target_t = generate_mq_coeffs(n=5, m=5, seed=7)
    set_random_seed(7)
    for _ in range(5 * 15):
        F16.random_element()
    x0_check = [F16_TO_INT[F16.random_element()] for _ in range(5)]
    assert validate_solution(5, polys_t, target_t, x0_check), \
        "planted solution does not satisfy generated system"
    print("  OK -- planted solution satisfies generated system")

    print("Testing generate_whipped_instance (planted solution)...")
    R, xs, eqs, target, kn, x0_int = generate_whipped_instance(n=6, m=4, o=2, k=2, seed=42)
    x0_gf = [INT_TO_F16[v] for v in x0_int]
    sub   = {xs[i]: x0_gf[i] for i in range(kn)}
    for a, eq in enumerate(eqs):
        val = eq.subs(sub)
        assert val == F16.zero(), f"Equation {a} not satisfied: got {val}"
    print("  OK -- planted solution satisfies all whipped equations")

    print("Testing reduce_with_planted_solution (solution survives reduction)...")
    eqs_red, feqs, free_idx = reduce_with_planted_solution(R, xs, eqs, kn, 4, x0_int, seed=42)
    free_sub = {xs[free_idx[i]]: x0_gf[free_idx[i]] for i in range(4)}
    for a, eq in enumerate(eqs_red):
        val = eq.subs(free_sub)
        assert val == F16.zero(), f"Reduced equation {a} not satisfied: got {val}"
    print("  OK -- planted solution satisfies all reduced equations")

    print("Testing generate_random_instance (planted solution)...")
    R2, ys, eqs2, target2, y0_int = generate_random_instance(kn=12, m=4, seed=99)
    y0_gf = [INT_TO_F16[v] for v in y0_int]
    sub2  = {ys[i]: y0_gf[i] for i in range(12)}
    for a, eq in enumerate(eqs2):
        val = eq.subs(sub2)
        assert val == F16.zero(), f"Random equation {a} not satisfied: got {val}"
    print("  OK -- planted solution satisfies all random equations")

    print("\nAll tests passed.")
