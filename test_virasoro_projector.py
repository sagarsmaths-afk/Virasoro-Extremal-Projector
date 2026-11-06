#!/usr/bin/env python3
"""
Unit tests for virasoro_projector.py verifying:
  1. Partition ordering and dimensions for n = 0..4.
  2. Symmetry of the Shapovalov matrices S^(n) and projector matrices B^{(n)}.
  3. Annihilation of M[-n] by the partial projector P_n = sum_{k=0}^n P[k] for 1 <= n <= 3.
  4. Agreement of the exact least common denominator D_N(L0, c) with prod_{m=1}^N tau_{-m}(K_m).
"""

import unittest
import sympy as sp

from virasoro_projector import (
    L0,
    action_of_Pk_on_level_n,
    compute_extremal_projector,
    kac_polynomial_Km,
    partitions_asc,
)


class TestVirasoroProjector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = compute_extremal_projector(max_level=3)

    def test_partition_counts(self):
        expected = {0: 1, 1: 1, 2: 2, 3: 3, 4: 5}
        for n, count in expected.items():
            self.assertEqual(len(partitions_asc(n)), count)

    def test_symmetry(self):
        for n in range(1, 4):
            S_n = self.data["S"][n]
            B_lf = self.data["B_left"][n]
            self.assertEqual(S_n, S_n.T)
            self.assertEqual(B_lf, B_lf.T)

    def test_annihilation_on_universal_verma(self):
        # Verify that sum_{k=0}^n P[k] annihilates M[-n] for n = 1, 2, 3
        for n in range(1, 4):
            dim_n = len(partitions_asc(n))
            total_action = sp.zeros(dim_n, dim_n)
            for k in range(n + 1):
                total_action += action_of_Pk_on_level_n(k, self.data["B_shifted"][k], n)
            total_action = total_action.applyfunc(sp.cancel)
            self.assertEqual(total_action, sp.zeros(dim_n, dim_n))

    def test_denominator_matches_shifted_kac_product(self):
        # Verify D_N(L0, c) == prod_{m=1}^N K_m(L0 - m, c) up to a nonzero rational constant
        for n in range(1, 4):
            prod_kac = sp.Integer(1)
            for m in range(1, n + 1):
                prod_kac *= kac_polynomial_Km(m).subs(L0, L0 - m)
            ratio = sp.cancel(self.data["LCD_PN"][n] / prod_kac)
            self.assertTrue(ratio.is_Rational and ratio != 0)


if __name__ == "__main__":
    unittest.main()
