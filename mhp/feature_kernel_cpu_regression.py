#!/usr/bin/env python3
"""Numerical regression checks for CPU feature-mode optimizations."""

from concurrent.futures import ThreadPoolExecutor
import unittest

import numpy as np

import mhp.em as em
import mhp.feature_kernel as feature_kernel
from mhp.em import (
    MHPConfig,
    _build_candidate_bloom,
    _lookup_candidate_rows,
    fit_mhp_feature,
)
from mhp.events import EventCollection
from mhp.feature_kernel import (
    _EPS,
    _q_grad_hess,
    _q_grad_hess_dynamic,
    _q_grad_hess_dynamic_coo,
    _q_grad_hess_dynamic_coo_parallel,
    _resolve_mstep_workers,
    fit_dynamic_weights_mstep_coo,
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

    def test_candidate_bloom_prefilter_preserves_exact_lookup(self):
        rng = np.random.default_rng(271)
        candidates = np.sort(
            rng.choice(2_000_000, size=12_000, replace=False).astype(np.int64)
        )
        pair_keys = rng.integers(0, 2_000_000, size=80_000, dtype=np.int64)
        pair_keys[:4_000] = rng.choice(candidates, size=4_000, replace=True)
        rng.shuffle(pair_keys)

        bloom = _build_candidate_bloom(candidates)
        self.assertIsNotNone(bloom)
        expected_rows, expected_idx = _lookup_candidate_rows(
            pair_keys, candidates, None
        )
        actual_rows, actual_idx = _lookup_candidate_rows(
            pair_keys, candidates, bloom
        )
        np.testing.assert_array_equal(actual_rows, expected_rows)
        np.testing.assert_array_equal(actual_idx, expected_idx)

        empty_rows, empty_idx = _lookup_candidate_rows(
            np.zeros(0, dtype=np.int64), candidates, bloom
        )
        self.assertEqual(empty_rows.size, 0)
        self.assertEqual(empty_idx.size, 0)

    def test_target_pair_cache_matches_streaming_estep(self):
        rng = np.random.default_rng(109)
        n_events = 72
        n_dims = 7
        times = np.cumsum(rng.uniform(0.03, 0.35, size=n_events))
        dims = rng.integers(0, n_dims, size=n_events, dtype=np.int64)
        events = EventCollection(
            times=times,
            dims=dims,
            M=n_dims,
            T=float(times[-1] + 0.5),
        )
        targets = np.repeat(np.arange(n_dims, dtype=np.int64), n_dims)
        sources = np.tile(np.arange(n_dims, dtype=np.int64), n_dims)
        candidates = targets.size
        phi = np.column_stack(
            [
                np.ones(candidates, dtype=np.float32),
                (targets == sources).astype(np.float32),
                ((targets + sources) % 3 == 0).astype(np.float32),
            ]
        )
        src_combo = np.zeros(n_events, dtype=np.int64)
        tgt_combo = rng.integers(0, 8, size=n_events, dtype=np.int64)
        combo_bits = (
            (np.arange(64, dtype=np.int64)[:, None] >> np.arange(6)) & 1
        ).astype(np.float64)
        exposure_rows = np.repeat(
            np.arange(candidates, dtype=np.int32), 8
        )
        exposure_combos = np.tile(np.arange(8, dtype=np.uint8), candidates)
        exposure_values = rng.uniform(
            0.25, 2.0, size=exposure_rows.size
        ).astype(np.float32)
        exposure = {
            "format": "coo",
            "shape": (candidates, 64),
            "rows": exposure_rows,
            "combos": exposure_combos,
            "values": exposure_values,
        }
        config = MHPConfig(
            history_window=2.5,
            max_history_events=16,
            max_iters=3,
            tol=0.0,
            beta_mode="shared",
            beta_shared_value=1.3,
            edge_threshold=0.0,
            chunk_size=13,
            estep_workers=2,
            mstep_workers=1,
            estep_device="cpu",
            verbose=False,
        )

        def run():
            return fit_mhp_feature(
                events,
                config,
                cand_targets=targets,
                cand_sources=sources,
                cand_phi=phi,
                feature_names=["bias", "self", "mod3"],
                src_combo=src_combo,
                tgt_combo=tgt_combo,
                dynamic_combo_bits=combo_bits,
                dynamic_exposure_2d=exposure,
                dynamic_feature_names=[f"state_{i}" for i in range(6)],
            )

        cached = run()
        old_limit = em._ESTEP_PAIR_CACHE_WORK_BYTES
        try:
            em._ESTEP_PAIR_CACHE_WORK_BYTES = 0
            streamed = run()
        finally:
            em._ESTEP_PAIR_CACHE_WORK_BYTES = old_limit

        np.testing.assert_array_equal(cached.p_self, streamed.p_self)
        np.testing.assert_array_equal(
            cached.feature_kernel.weights, streamed.feature_kernel.weights
        )
        np.testing.assert_array_equal(
            cached.params.edge_alpha, streamed.params.edge_alpha
        )
        self.assertEqual(cached.log_likelihood, streamed.log_likelihood)
        self.assertEqual(cached.iterations_run, streamed.iterations_run)
        for cached_entry, streamed_entry in zip(cached.trace, streamed.trace):
            for key in (
                "log_likelihood",
                "delta_rel",
                "p_self_mean",
                "alpha_max",
                "alpha_median",
                "spectral_radius",
                "parameter_delta_rel",
                "convergence_delta_rel",
            ):
                self.assertEqual(cached_entry[key], streamed_entry[key])

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

    def test_parallel_coo_mstep_matches_serial_float64_path(self):
        rng = np.random.default_rng(91)
        n_w = self.features + self.dynamic_features
        w = rng.normal(0.0, 0.1, n_w)
        w0 = np.zeros(n_w)
        reg = np.ones(n_w)
        reg[0] = 0.0

        def make_coo(nnz):
            return (
                rng.integers(0, self.candidates, nnz, dtype=np.int32),
                rng.integers(0, self.combos, nnz, dtype=np.uint8),
                rng.random(nnz, dtype=np.float32),
            )

        n_coo = make_coo(20_000)
        e_coo = make_coo(35_000)
        l2 = 0.02
        serial = _q_grad_hess_dynamic_coo(
            w, self.phi, self.bits, n_coo, e_coo, l2, w0, reg
        )
        with ThreadPoolExecutor(max_workers=3) as executor:
            parallel = _q_grad_hess_dynamic_coo_parallel(
                w,
                self.phi,
                self.bits,
                n_coo,
                e_coo,
                l2,
                w0,
                reg,
                executor,
            )
        self.assertEqual(parallel[0], serial[0])
        np.testing.assert_allclose(parallel[1], serial[1], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(parallel[2], serial[2], rtol=0.0, atol=0.0)

        fitted_serial = fit_dynamic_weights_mstep_coo(
            self.phi,
            self.bits,
            n_coo,
            e_coo,
            w,
            l2=l2,
            max_iter=20,
            mstep_workers=1,
        )
        fitted_parallel = fit_dynamic_weights_mstep_coo(
            self.phi,
            self.bits,
            n_coo,
            e_coo,
            w,
            l2=l2,
            max_iter=20,
            mstep_workers=3,
        )
        np.testing.assert_allclose(
            fitted_parallel, fitted_serial, rtol=0.0, atol=2e-14
        )

        empty_n = (
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.uint8),
            np.zeros(0, dtype=np.float32),
        )
        serial_empty_n = _q_grad_hess_dynamic_coo(
            w, self.phi, self.bits, empty_n, e_coo, l2, w0, reg
        )
        with ThreadPoolExecutor(max_workers=3) as executor:
            parallel_empty_n = _q_grad_hess_dynamic_coo_parallel(
                w,
                self.phi,
                self.bits,
                empty_n,
                e_coo,
                l2,
                w0,
                reg,
                executor,
            )
        self.assertEqual(parallel_empty_n[0], serial_empty_n[0])
        np.testing.assert_allclose(
            parallel_empty_n[1], serial_empty_n[1], rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            parallel_empty_n[2], serial_empty_n[2], rtol=0.0, atol=0.0
        )

        nonbinary_bits = self.bits.copy()
        nonbinary_bits[:, 0] *= 0.25
        serial_nonbinary = _q_grad_hess_dynamic_coo(
            w, self.phi, nonbinary_bits, n_coo, e_coo, l2, w0, reg
        )
        with ThreadPoolExecutor(max_workers=3) as executor:
            parallel_nonbinary = _q_grad_hess_dynamic_coo_parallel(
                w,
                self.phi,
                nonbinary_bits,
                n_coo,
                e_coo,
                l2,
                w0,
                reg,
                executor,
            )
        self.assertEqual(parallel_nonbinary[0], serial_nonbinary[0])
        np.testing.assert_allclose(
            parallel_nonbinary[1], serial_nonbinary[1], rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            parallel_nonbinary[2], serial_nonbinary[2], rtol=0.0, atol=0.0
        )

    def test_parallel_worker_memory_budget_includes_blocks_and_coo_entries(self):
        self.assertEqual(
            _resolve_mstep_workers(
                6, 400_000, 6, 125, 800_000, 2_400_000, 1
            ),
            6,
        )
        self.assertEqual(
            _resolve_mstep_workers(
                6, 700_000, 6, 125, 1_400_000, 4_200_000, 1
            ),
            1,
        )
        self.assertEqual(
            _resolve_mstep_workers(
                6, 400_000, 6, 125, 4_000_000, 12_000_000, 1
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
