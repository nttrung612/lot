import numpy as np

from lot_experiments.backups import lot_backup
from lot_experiments.pruning import admissible_omitted_mass, sharp_pruning_error, top_mass_mask


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

