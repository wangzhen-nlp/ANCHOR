#!/usr/bin/env python3
"""Numerical regression checks for the cache-blocked feature M-step math."""

import unittest

import numpy as np

import mhp.feature_kernel as feature_kernel
from mhp.feature_kernel import (
    _EPS,
    _q_grad_hess,
    _q_grad_hess_dynamic,
    _q_grad_hess_dynamic_coo,
    sigmoid,
    softplus,
    softplus_sigmoid,
)


def _explicit_dynamic(w, phi, bits, n2d, e2d, l2, w0, reg_mask):
    """Dense reference using the fully materialized (candidate, combo) table."""
    f_static = phi.shape[1]
    z = phi.astype(np.float64) @ w[:f_static, None]
    z = z + (bits @ w[f_static:])[None, :]
    alpha = softplus(z)
    alpha_safe = np.maximum(alpha, _EPS)
    sig = sigmoid(z)
    q = float(
        np.sum(n2d * np.log(alpha_safe) - e2d * alpha)
        - l2 * np.sum(reg_mask * (w - w0) ** 2)
    )
    coef = (n2d / alpha_safe - e2d) * sig
    h = (
        (n2d / alpha_safe - e2d) * sig * (1.0 - sig)
        - n2d * sig * sig / (alpha_safe * alpha_safe)
    )
    grad_s = phi.T @ coef.sum(axis=1)
    grad_d = bits.T @ coef.sum(axis=0)
    grad = np.concatenate([grad_s, grad_d]) - 2.0 * l2 * reg_mask * (w - w0)
    hss = phi.T @ (phi * h.sum(axis=1, keepdims=True))
    hsd = phi.T @ (h @ bits)
    hdd = bits.T @ (bits * h.sum(axis=0)[:, None])
    hess = np.block([[hss, hsd], [hsd.T, hdd]])
    hess[np.arange(len(w)), np.arange(len(w))] -= 2.0 * l2 * reg_mask
    return q, grad, hess


class FeatureKernelCpuOptimizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(19)
        # More than _MATMUL_BLOCK_ROWS so every mixed float32/float64 blocked
        # path is exercised, while remaining small enough for fast unit tests.
        cls.candidates = 5003
        cls.features = 7
        cls.combos = 8
        cls.dynamic_features = 3
        cls.phi = (rng.random((cls.candidates, cls.features)) < 0.28).astype(
            np.float32
        )
        cls.phi[:, 0] = 1.0
        cls.bits = (
            (np.arange(cls.combos)[:, None] >> np.arange(cls.dynamic_features)) & 1
        ).astype(np.float64)
        cls.rng = rng

    def test_fused_softplus_sigmoid_is_stable_at_both_tails(self):
        z = np.array([-1000.0, -50.0, -1.0, 0.0, 1.0, 50.0, 1000.0])
        actual_a, actual_s = softplus_sigmoid(z)
        np.testing.assert_allclose(actual_a, softplus(z), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(actual_s, sigmoid(z), rtol=2e-15, atol=0.0)

    def test_static_q_gradient_hessian_matches_direct_formula(self):
        rng = self.rng
        phi = self.phi
        f = self.features
        w = rng.normal(0.0, 0.25, f)
        w0 = rng.normal(0.0, 0.1, f)
        reg = np.ones(f)
        reg[0] = 0.0
        n_resp = rng.random(self.candidates)
        exposure = 0.2 + rng.random(self.candidates)
        l2 = 0.03

        actual = _q_grad_hess(w, phi, n_resp, exposure, l2, w0, reg)
        z = phi.astype(np.float64) @ w
        alpha = softplus(z)
        sig = sigmoid(z)
        coef = (n_resp / alpha - exposure) * sig
        h = (
            (n_resp / alpha - exposure) * sig * (1.0 - sig)
            - n_resp * sig * sig / (alpha * alpha)
        )
        q = float(
            np.sum(n_resp * np.log(alpha) - exposure * alpha)
            - l2 * np.sum(reg * (w - w0) ** 2)
        )
        grad = phi.T @ coef - 2.0 * l2 * reg * (w - w0)
        hess = phi.T @ (phi * h[:, None])
        hess[np.arange(f), np.arange(f)] -= 2.0 * l2 * reg

        self.assertAlmostEqual(actual[0], q, places=10)
        np.testing.assert_allclose(actual[1], grad, rtol=2e-12, atol=2e-10)
        np.testing.assert_allclose(actual[2], hess, rtol=2e-12, atol=2e-10)

    def test_dense_dynamic_q_gradient_hessian_matches_explicit_table(self):
        rng = self.rng
        n_w = self.features + self.dynamic_features
        w = rng.normal(0.0, 0.2, n_w)
        w0 = rng.normal(0.0, 0.05, n_w)
        reg = np.ones(n_w)
        reg[0] = 0.0
        n2d = rng.random((self.candidates, self.combos))
        e2d = 0.1 + rng.random((self.candidates, self.combos))
        l2 = 0.02

        actual = _q_grad_hess_dynamic(
            w, self.phi, self.bits, n2d, e2d, l2, w0, reg
        )
        expected = _explicit_dynamic(
            w, self.phi, self.bits, n2d, e2d, l2, w0, reg
        )
        self.assertAlmostEqual(actual[0], expected[0], places=9)
        np.testing.assert_allclose(actual[1], expected[1], rtol=3e-12, atol=3e-9)
        np.testing.assert_allclose(actual[2], expected[2], rtol=3e-12, atol=3e-9)

        old_cap = feature_kernel._DYNAMIC_CROSS_WORK_BYTES
        feature_kernel._DYNAMIC_CROSS_WORK_BYTES = 1
        try:
            low_memory = _q_grad_hess_dynamic(
                w, self.phi, self.bits, n2d, e2d, l2, w0, reg
            )
        finally:
            feature_kernel._DYNAMIC_CROSS_WORK_BYTES = old_cap
        self.assertAlmostEqual(low_memory[0], expected[0], places=9)
        np.testing.assert_allclose(low_memory[1], expected[1], rtol=3e-12, atol=3e-9)
        np.testing.assert_allclose(low_memory[2], expected[2], rtol=3e-12, atol=3e-9)

    def test_coo_dynamic_q_gradient_hessian_matches_explicit_table(self):
        rng = self.rng
        n_w = self.features + self.dynamic_features
        w = rng.normal(0.0, 0.2, n_w)
        w0 = rng.normal(0.0, 0.05, n_w)
        reg = np.ones(n_w)
        reg[0] = 0.0

        def make_coo(nnz):
            return (
                rng.integers(0, self.candidates, nnz, dtype=np.int32),
                rng.integers(0, self.combos, nnz, dtype=np.uint8),
                rng.random(nnz, dtype=np.float32),
            )

        n_coo = make_coo(18_000)
        e_coo = make_coo(31_000)
        n2d = np.zeros((self.candidates, self.combos))
        e2d = np.zeros_like(n2d)
        np.add.at(n2d, (n_coo[0], n_coo[1]), n_coo[2])
        np.add.at(e2d, (e_coo[0], e_coo[1]), e_coo[2])
        l2 = 0.015

        actual = _q_grad_hess_dynamic_coo(
            w, self.phi, self.bits, n_coo, e_coo, l2, w0, reg
        )
        expected = _explicit_dynamic(
            w, self.phi, self.bits, n2d, e2d, l2, w0, reg
        )
        self.assertAlmostEqual(actual[0], expected[0], places=8)
        np.testing.assert_allclose(actual[1], expected[1], rtol=5e-12, atol=5e-9)
        np.testing.assert_allclose(actual[2], expected[2], rtol=5e-12, atol=5e-9)

        # Force the memory-capped fallback without allocating millions of rows.
        old_cap = feature_kernel._DYNAMIC_CROSS_WORK_BYTES
        feature_kernel._DYNAMIC_CROSS_WORK_BYTES = 1
        try:
            low_memory = _q_grad_hess_dynamic_coo(
                w, self.phi, self.bits, n_coo, e_coo, l2, w0, reg
            )
        finally:
            feature_kernel._DYNAMIC_CROSS_WORK_BYTES = old_cap
        self.assertAlmostEqual(low_memory[0], expected[0], places=8)
        np.testing.assert_allclose(low_memory[1], expected[1], rtol=5e-12, atol=5e-9)
        np.testing.assert_allclose(low_memory[2], expected[2], rtol=5e-12, atol=5e-9)


if __name__ == "__main__":
    unittest.main()
