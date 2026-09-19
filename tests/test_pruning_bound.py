import numpy as np

from lot_experiments.backups import lot_backup
from lot_experiments.pruning import (
    admissible_omitted_mass,
    poisson_backup_error_bound,
    sharp_pruning_error,
    sharp_pruning_error_from_masses,
    top_mass_mask,
)


def test_sharp_pruning_construction_attains_closed_form_error():
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    retained = np.array([True, True, False, False])
    span = 0.8
    T0 = 0.2
    q = np.where(retained, 0.0, span)
    full = lot_backup(q, weights[:, None], [1.0], T0)
    pruned = lot_backup(q, weights[:, None], [1.0], T0, retained=retained[:, None])
    alpha = float(weights[~retained].sum())
    observed = full.value - pruned.value
    expected = sharp_pruning_error(alpha, span, T0)
    np.testing.assert_allclose(observed, expected, atol=1e-14, rtol=0.0)
    assert observed >= 0.0


def test_top_mass_mask_is_minimum_prefix_and_keeps_original_weights():
    weights = np.array([0.36, 0.29, 0.2, 0.1, 0.05])
    mask = top_mass_mask(weights, alpha=0.2)
    assert mask.sum() == 3
    assert weights[mask].sum() >= 0.8
    assert np.sort(weights)[-2:].sum() < 0.8


def test_admissible_mass_inverts_sharp_error_formula():
    span = 1.1
    tolerance = 0.07
    T0 = 0.25
    alpha = admissible_omitted_mass(span, tolerance, T0)
    np.testing.assert_allclose(
        sharp_pruning_error(alpha, span, T0), tolerance, atol=2e-16, rtol=0.0
    )


def test_poisson_backup_bound_uses_the_heat_mixture_tail():
    beta_r = 0.03
    span = 0.8
    T0 = 0.2
    expected = T0 * np.log(1.0 + beta_r * np.expm1(span / T0))
    np.testing.assert_allclose(
        poisson_backup_error_bound(beta_r, span, T0),
        expected,
        atol=2e-16,
        rtol=0.0,
    )
    assert poisson_backup_error_bound(0.0, span, T0) == 0.0
    assert poisson_backup_error_bound(1.0, span, T0) == span


def test_sharp_formula_remains_stable_for_tiny_retained_mass():
    retained = 1e-20
    omitted = 1.0
    observed = sharp_pruning_error_from_masses(retained, omitted, span=0.8, T0=0.2)
    assert np.isfinite(observed)
    np.testing.assert_allclose(observed, 0.8 - 0.2 * np.log(retained), rtol=1e-12)
