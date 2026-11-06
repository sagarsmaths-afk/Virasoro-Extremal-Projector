#!/usr/bin/env python3
"""
virasoro_projector.py
=====================
Exact symbolic computation of the Virasoro extremal projector components P[n],
the partial projectors P_N = sum_{n=0}^N P[n], the Shapovalov matrices S^(n),
the primitive Kac polynomials K_n(L0, c), and the exact N-th regularizing
denominators D_N(L0, c).

Mathematical Conventions
------------------------
- Virasoro Lie algebra commutation relations:
    [L_m, L_n] = (m - n) L_{m+n} + ((m^3 - m) / 12) * delta_{m+n, 0} * c
    [L_0, L_{-k}] = k L_{-k}   ==>   f(L_0) L_{-k} = L_{-k} f(L_0 + k)

- For an integer partition lambda = (lambda_1 <= lambda_2 <= ... <= lambda_k) of n:
    L_lambda        = L_{lambda_1} L_{lambda_2} ... L_{lambda_k}    (lowers level by n)
    L_lambda^dagger = L_{-lambda_k} ... L_{-lambda_2} L_{-lambda_1} (raises level by n)

- Level-n component P[n] in left-placement normal-ordered form:
    P[n] = sum_{lambda, mu |- n} b^{(n)}_{lambda, mu}(L_0, c) * L_lambda^dagger * L_mu
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from typing import Dict, List, Tuple

import sympy as sp
from sympy.polys.matrices import DomainMatrix

__all__ = [
    "L0",
    "c",
    "partitions_asc",
    "format_partition",
    "apply_Lm_to_basis",
    "apply_L_partition_pos",
    "apply_L_partition_neg",
    "shapovalov_matrix",
    "action_of_Pk_on_level_n",
    "compute_extremal_projector",
    "kac_polynomial_Km",
    "print_report",
]

# Global symbolic variables for the Cartan generator L_0 and central charge c
L0, c = sp.symbols("L0 c")

Partition = Tuple[int, ...]
StateDict = Dict[Partition, sp.Expr]


# ---------------------------------------------------------------------------
# Fast Exact Linear Algebra over Q[L0, c] / Q(L0, c) via SymPy DomainMatrix
# ---------------------------------------------------------------------------
def fast_det(M: sp.Matrix) -> sp.Expr:
    """Compute the exact factored determinant of a symbolic matrix using DomainMatrix."""
    dm = DomainMatrix.from_Matrix(M)
    return sp.factor(dm.det().as_expr())


def fast_solve_right(rhs_mat: sp.Matrix, S_n: sp.Matrix) -> sp.Matrix:
    """Compute rhs_mat * S_n^{-1} exactly over Q(L0, c) using DomainMatrix."""
    dm_rhs = DomainMatrix.from_Matrix(rhs_mat).to_field()
    dm_S = DomainMatrix.from_Matrix(S_n).to_field()
    dm_res = dm_rhs * dm_S.inv()
    return dm_res.to_Matrix()


# ---------------------------------------------------------------------------
# 1. Integer Partitions of n in Ascending Order
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def partitions_asc(n: int, min_part: int = 1) -> List[Partition]:
    """
    Return all partitions of n as ascending tuples (lambda_1 <= ... <= lambda_k),
    ordered from finest (1^n) to coarsest (n).
    """
    if n == 0:
        return [()]
    result: List[Partition] = []
    for first in range(min_part, n + 1):
        for rest in partitions_asc(n - first, first):
            result.append((first,) + rest)
    result.sort(key=lambda p: (-len(p), tuple(reversed(p))))
    return result


def format_partition(lam: Partition) -> str:
    """Format a partition tuple such as (1, 1, 1) -> '(1^3)', (1, 2) -> '(1,2)'."""
    if not lam:
        return "(0)"
    counts: Dict[int, int] = {}
    for x in lam:
        counts[x] = counts.get(x, 0) + 1
    parts = [f"{x}" if cnt == 1 else f"{x}^{cnt}" for x, cnt in sorted(counts.items())]
    return "(" + ",".join(parts) + ")"


# ---------------------------------------------------------------------------
# 2. Action of Virasoro Generators L_m on the Universal Verma Module M
# ---------------------------------------------------------------------------
def _add_state(state: StateDict, lam: Partition, expr: sp.Expr) -> None:
    """Accumulate a term `L_lam^dagger * expr |1>` into `state`."""
    if expr == 0:
        return
    state[lam] = state.get(lam, sp.Integer(0)) + expr


@lru_cache(maxsize=None)
def apply_Lm_to_basis(m: int, lam: Partition) -> StateDict:
    """
    Compute L_m (L_lam^dagger |1>) in the right-coefficient basis of the universal
    Verma module M:
        sum_{mu} L_mu^dagger * coeff_mu(L0, c) |1>,
    where L_0 |1> = L0 |1> and L_k |1> = 0 for k > 0.
    """
    if not lam:
        if m > 0:
            return {}
        if m == 0:
            return {(): L0}
        return {(-m,): sp.Integer(1)}

    # Write L_lam^dagger = L_{-a} * L_{rest}^dagger where a = lam[-1] is the largest part
    a = lam[-1]
    rest = lam[:-1]

    # Case 1: m < 0 and (-m) >= a -> already in PBW normal order
    if m < 0 and (-m) >= a:
        return {lam + (-m,): sp.Integer(1)}

    # Case 2: Commute L_m past L_{-a}:
    #   L_m L_{-a} = L_{-a} L_m + (m + a) L_{m - a} + delta_{m, a} * ((m^3 - m)/12) * c
    res: StateDict = {}

    # Term 2a: L_{-a} (L_m L_{rest}^dagger |1>)
    for mu, coeff in apply_Lm_to_basis(m, rest).items():
        for nu, c_nu in apply_Lm_to_basis(-a, mu).items():
            _add_state(res, nu, c_nu * coeff)

    # Term 2b: (m + a) L_{m - a} L_{rest}^dagger |1>
    if m + a != 0:
        for mu, coeff in apply_Lm_to_basis(m - a, rest).items():
            _add_state(res, mu, (m + a) * coeff)

    # Term 2c: central charge term when m == a
    if m == a:
        central_factor = sp.Rational(m**3 - m, 12) * c
        if central_factor != 0:
            _add_state(res, rest, central_factor)

    return {k: sp.expand(v) for k, v in res.items() if v != 0}


def apply_L_partition_pos(mu: Partition, lam: Partition) -> StateDict:
    """
    Compute L_mu (L_lam^dagger |1>) where mu = (mu_1 <= ... <= mu_r) and
    L_mu = L_{mu_1} L_{mu_2} ... L_{mu_r}.
    """
    state: StateDict = {lam: sp.Integer(1)}
    for m in reversed(mu):
        next_state: StateDict = {}
        for nu, coeff in state.items():
            for rho, c_rho in apply_Lm_to_basis(m, nu).items():
                _add_state(next_state, rho, c_rho * coeff)
        state = {k: sp.expand(v) for k, v in next_state.items() if v != 0}
    return state


def apply_L_partition_neg(mu: Partition, lam: Partition) -> StateDict:
    """
    Compute L_mu^dagger (L_lam^dagger |1>) where mu = (mu_1 <= ... <= mu_r) and
    L_mu^dagger = L_{-mu_r} ... L_{-mu_1}.
    """
    state: StateDict = {lam: sp.Integer(1)}
    for m in mu:
        next_state: StateDict = {}
        for nu, coeff in state.items():
            for rho, c_rho in apply_Lm_to_basis(-m, nu).items():
                _add_state(next_state, rho, c_rho * coeff)
        state = {k: sp.expand(v) for k, v in next_state.items() if v != 0}
    return state


# ---------------------------------------------------------------------------
# 3. Shapovalov Matrix S^(n)(L0, c) at Level n
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def shapovalov_matrix(n: int) -> sp.Matrix:
    """
    Compute the level-n Virasoro Shapovalov matrix S^(n)(L0, c) indexed by
    partitions of n, defined by:
        L_mu L_eta^dagger |1> = S^(n)_{mu, eta}(L0, c) |1>.
    """
    parts = partitions_asc(n)
    dim = len(parts)
    S = sp.zeros(dim, dim)
    for i, mu in enumerate(parts):
        for j, eta in enumerate(parts):
            res = apply_L_partition_pos(mu, eta)
            S[i, j] = sp.expand(res.get((), sp.Integer(0)))
    return S


# ---------------------------------------------------------------------------
# 4. Action of P[k] on M[-n] and Recursive Computation of P[n]
# ---------------------------------------------------------------------------
def action_of_Pk_on_level_n(k: int, B_k_shifted: sp.Matrix, n: int) -> sp.Matrix:
    """
    Compute the matrix representing the action of P[k] on the weight space M[-n]
    in the right-coefficient basis {L_lam^dagger |1>}_{lam |- n}.
    """
    parts_n = partitions_asc(n)
    dim_n = len(parts_n)
    idx_n = {p: idx for idx, p in enumerate(parts_n)}

    if k == 0:
        return sp.eye(dim_n)
    if k > n:
        return sp.zeros(dim_n, dim_n)

    parts_k = partitions_asc(k)
    dim_k = len(parts_k)

    # Shift B_k_shifted(L0, c) = b^{(k)}(L0 + k, c) by L0 -> L0 + (n - k)
    shift = n - k
    B_at_n = B_k_shifted.subs(L0, L0 + shift) if shift != 0 else B_k_shifted

    M_act = sp.zeros(dim_n, dim_n)

    for j, eta in enumerate(parts_n):
        L_beta_on_eta = [apply_L_partition_pos(beta, eta) for beta in parts_k]

        for a_idx, alpha in enumerate(parts_k):
            combined_mid: StateDict = {}
            for b_idx in range(dim_k):
                b_coeff = B_at_n[a_idx, b_idx]
                if b_coeff == 0:
                    continue
                for rho, r_coeff in L_beta_on_eta[b_idx].items():
                    _add_state(combined_mid, rho, b_coeff * r_coeff)

            for rho, mid_coeff in combined_mid.items():
                if mid_coeff == 0:
                    continue
                for lam, l_coeff in apply_L_partition_neg(alpha, rho).items():
                    i = idx_n[lam]
                    M_act[i, j] += l_coeff * mid_coeff

    return M_act.applyfunc(sp.cancel)


def compute_extremal_projector(max_level: int) -> Dict[str, dict]:
    """
    Recursively compute the Virasoro extremal projector components P[0], ..., P[max_level].

    Returns a dictionary containing:
      - 'partitions':  {n: list of ascending partitions of n}
      - 'S':           {n: Shapovalov matrix S^(n)(L0, c)}
      - 'B_shifted':   {n: matrix b^{(n)}_{lam, mu}(L0 + n, c)}
      - 'B_left':      {n: matrix b^{(n)}_{lam, mu}(L0, c) in left-placement form}
      - 'LCD_Pn_part': {n: least common denominator of P[n] in left-placement form}
      - 'LCD_PN':      {n: least common denominator D_n(L0, c) of P_n = sum_{k=0}^n P[k]}
    """
    partitions: Dict[int, List[Partition]] = {0: [()]}
    S_mats: Dict[int, sp.Matrix] = {}
    B_shifted: Dict[int, sp.Matrix] = {0: sp.Matrix([[sp.Integer(1)]])}
    B_left: Dict[int, sp.Matrix] = {0: sp.Matrix([[sp.Integer(1)]])}
    LCD_part: Dict[int, sp.Expr] = {0: sp.Integer(1)}
    LCD_total: Dict[int, sp.Expr] = {0: sp.Integer(1)}

    for n in range(1, max_level + 1):
        parts_n = partitions_asc(n)
        partitions[n] = parts_n
        dim_n = len(parts_n)

        # 1. Shapovalov matrix S^(n)(L0, c)
        S_n = shapovalov_matrix(n)
        S_mats[n] = S_n

        # 2. Sum the action of -P[0], -P[1], ..., -P[n-1] on M[-n]
        rhs_mat = sp.zeros(dim_n, dim_n)
        for k in range(n):
            rhs_mat -= action_of_Pk_on_level_n(k, B_shifted[k], n)
        rhs_mat = rhs_mat.applyfunc(sp.cancel)

        # 3. Solve B_n_shifted * S_n = rhs_mat  ==>  B_n_shifted = rhs_mat * S_n^{-1}
        B_n_sh = fast_solve_right(rhs_mat, S_n).applyfunc(sp.factor)
        B_shifted[n] = B_n_sh

        # 4. Left-placement form: replace L0 -> L0 - n
        B_n_lf = B_n_sh.subs(L0, L0 - n).applyfunc(sp.factor)
        B_left[n] = B_n_lf

        # 5. Compute the exact least common denominator (LCD) of P[n] and of P_n
        denoms_n = []
        for entry in B_n_lf:
            num, den = sp.fraction(sp.cancel(entry))
            if num != 0:
                denoms_n.append(sp.Poly(den, L0, c, domain="QQ"))
        lcd_n_poly = sp.Integer(1)
        for d in denoms_n:
            lcd_n_poly = sp.lcm(lcd_n_poly, d.as_expr())
        _, lcd_n_prim = sp.primitive(sp.expand(lcd_n_poly))
        LCD_part[n] = sp.factor(lcd_n_prim)

        lcd_tot_poly = sp.lcm(sp.expand(LCD_total[n - 1]), sp.expand(lcd_n_prim))
        _, lcd_tot_prim = sp.primitive(sp.expand(lcd_tot_poly))
        LCD_total[n] = sp.factor(lcd_tot_prim)

    return {
        "partitions": partitions,
        "S": S_mats,
        "B_shifted": B_shifted,
        "B_left": B_left,
        "LCD_Pn_part": LCD_part,
        "LCD_PN": LCD_total,
    }


# ---------------------------------------------------------------------------
# 5. Primitive Kac Polynomials K_m(L0, c) and Denominator Reporting
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def kac_polynomial_Km(m: int) -> sp.Expr:
    """
    Compute the primitive level-m Kac polynomial:
        K_m(L0, c) = prod_{r*s = m} psi_{r,s}(L0, c)
    via the recursive determinant quotient det(S^(m)) / prod_{k < m} K_k^{p(m-k)}.
    """
    S_m = shapovalov_matrix(m)
    rem = fast_det(S_m)
    for k in range(1, m):
        p_mk = len(partitions_asc(m - k))
        K_k = kac_polynomial_Km(k)
        rem = sp.cancel(rem / (K_k**p_mk))
    _, prim = sp.primitive(sp.expand(rem))
    poly = sp.Poly(prim, L0, c)
    if poly.LC() < 0:
        prim = -prim
    return sp.factor(prim)


def print_report(max_level: int = 3, latex_only: bool = False) -> Dict[str, dict]:
    """Compute and print a full symbolic report up to `max_level`."""
    data = compute_extremal_projector(max_level)

    if not latex_only:
        print("=" * 78)
        print(f" VIRASORO EXTREMAL PROJECTOR COMPUTATION UP TO LEVEL N = {max_level}")
        print("=" * 78)

        for n in range(1, max_level + 1):
            parts = data["partitions"][n]
            p_names = [format_partition(p) for p in parts]
            S_n = data["S"][n]
            B_sh = data["B_shifted"][n]
            B_lf = data["B_left"][n]

            print(f"\n{'-' * 78}")
            print(f" LEVEL n = {n}   |   Ordered partitions: {', '.join(p_names)}")
            print(f"{'-' * 78}")
            print(f"det S^({n})(L0, c) = {fast_det(S_n)}")
            print(f"K_{n}(L0, c)       = {kac_polynomial_Km(n)}")
            print(f"Symmetric matrix?  {B_lf == B_lf.T}")

            print(f"\n[A] Shifted Coefficients b^({n})_{{lambda, mu}}(L0 + {n}, c):")
            for i in range(len(parts)):
                for j in range(i, len(parts)):
                    label = f"b_{{ {p_names[i]}, {p_names[j]} }}(L0 + {n}, c)"
                    if i != j:
                        label += f" = b_{{ {p_names[j]}, {p_names[i]} }}(L0 + {n}, c)"
                    print(f"  {label} = {B_sh[i, j]}")

            print(f"\n[B] Left-Placement Coefficients b^({n})_{{lambda, mu}}(L0, c):")
            for i in range(len(parts)):
                for j in range(i, len(parts)):
                    label = f"b_{{ {p_names[i]}, {p_names[j]} }}(L0, c)"
                    if i != j:
                        label += f" = b_{{ {p_names[j]}, {p_names[i]} }}(L0, c)"
                    print(f"  {label} = {B_lf[i, j]}")

            print(f"\n[C] Exact N-th Denominator Analysis at N = {n}:")
            print(f"  Least Common Denominator of P[{n}]:")
            print(f"    LCD(P[{n}]) = {data['LCD_Pn_part'][n]}")
            print(f"  Least Common Denominator of Partial Projector P_{n} = sum_{{k=0}}^{n} P[k]:")
            print(f"    LCD(P_{n})  = {data['LCD_PN'][n]}")

            single_prod = sp.Integer(1)
            for k in range(1, n + 1):
                K_k = kac_polynomial_Km(k)
                single_prod *= K_k.subs(L0, L0 - k)
            _, single_prim = sp.primitive(sp.expand(single_prod))
            print(f"  Product         prod_{{m=1}}^{n} tau_{{-m}}(K_m) = {sp.factor(single_prim)}")

    print(f"\n{'=' * 78}")
    print(f" LATEX SUMMARY OF OBTAINED DENOMINATORS (N = 1 .. {max_level})")
    print(f"{'=' * 78}")
    print("\\begin{align*}")
    for n in range(1, max_level + 1):
        denom_latex = sp.latex(data["LCD_PN"][n], symbol_names={L0: "L_0"})
        end_line = " \\\\[4pt]" if n < max_level else ""
        print(f"D_{{{n}}}(L_0, c) &= {denom_latex}{end_line}")
    print("\\end{align*}")

    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the Virasoro extremal projector P_N and its exact N-th denominator D_N(L0, c)."
    )
    parser.add_argument(
        "--max-level",
        type=int,
        default=3,
        help="Maximum level N to compute (default: 3)",
    )
    parser.add_argument(
        "--latex-only",
        action="store_true",
        help="Print only the final LaTeX summary of the denominators D_N(L0, c)",
    )
    # Use parse_known_args() for seamless execution inside Jupyter / IPython kernels
    args, _ = parser.parse_known_args()
    print_report(max_level=args.max_level, latex_only=args.latex_only)


if __name__ == "__main__":
    main()
