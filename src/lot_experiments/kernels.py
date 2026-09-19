"""Exact heat and normalized finite-walk heat reference kernels."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from scipy.special import gammaln, logsumexp
from scipy.stats import poisson


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class HeatApproximationCertificate:
    """Observed exact-vs-truncated error and the Poisson-tail certificate."""

    column_l1_errors: FloatArray
    max_column_l1_error: float
    beta_r: float
    l1_bound: float
    tolerance: float
    holds: bool


def _as_square_laplacian(
    laplacian: ArrayLike | sparse.spmatrix, *, tolerance: float = 1e-12
) -> sparse.csr_matrix:
    matrix = sparse.csr_matrix(laplacian, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("laplacian must be nonempty and square")
    if matrix.nnz and not np.all(np.isfinite(matrix.data)):
        raise ValueError("laplacian entries must be finite")
    difference = matrix - matrix.T
    if difference.nnz and np.max(np.abs(difference.data)) > tolerance:
        raise ValueError("laplacian must be symmetric")
    diagonal = matrix.diagonal()
    if np.min(diagonal, initial=0.0) < -tolerance:
        raise ValueError("laplacian diagonal must be nonnegative")
    off_diagonal = matrix.copy()
    off_diagonal.setdiag(0.0)
    off_diagonal.eliminate_zeros()
    if off_diagonal.nnz and np.max(off_diagonal.data) > tolerance:
        raise ValueError("laplacian off-diagonal entries must be nonpositive")
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    scale = max(1.0, float(np.max(np.abs(diagonal), initial=0.0)))
    if np.max(np.abs(row_sums), initial=0.0) > tolerance * scale:
        raise ValueError("laplacian rows must sum to zero")
    return matrix


def _clean_probability_columns(values: FloatArray, tolerance: float = 1e-12) -> FloatArray:
    cleaned = np.asarray(values, dtype=np.float64).copy()
    if np.min(cleaned, initial=0.0) < -tolerance:
        raise FloatingPointError("kernel has materially negative entries")
    cleaned[cleaned < 0.0] = 0.0
    totals = cleaned.sum(axis=0, keepdims=True)
    if np.any(totals <= 0.0):
        raise FloatingPointError("kernel has a zero-mass column")
    return cleaned / totals


def exact_heat_kernel(
    laplacian: ArrayLike | sparse.spmatrix,
    diffusion_time: float,
    *,
    tolerance: float = 1e-12,
) -> FloatArray:
    """Dense small-graph reference ``exp(-t L)``.

    This function intentionally forms a dense matrix and is only the validation
    reference. Local algorithms use :mod:`lot_experiments.heat_local`.
    """

    if diffusion_time < 0 or not math.isfinite(diffusion_time):
        raise ValueError("diffusion_time must be finite and nonnegative")
    matrix = _as_square_laplacian(laplacian)
    heat = expm(-diffusion_time * matrix.toarray())
    return _clean_probability_columns(heat, tolerance)


def exact_heat_column(
    laplacian: ArrayLike | sparse.spmatrix,
    diffusion_time: float,
    anchor: int,
    *,
    tolerance: float = 1e-12,
) -> FloatArray:
    """Compute one exact heat column without materializing the full heat matrix."""

    if diffusion_time < 0 or not math.isfinite(diffusion_time):
        raise ValueError("diffusion_time must be finite and nonnegative")
    matrix = _as_square_laplacian(laplacian)
    if not 0 <= anchor < matrix.shape[0]:
        raise IndexError(anchor)
    basis = np.zeros(matrix.shape[0], dtype=np.float64)
    basis[anchor] = 1.0
    column = np.asarray(expm_multiply(-diffusion_time * matrix, basis), dtype=np.float64)
    return _clean_probability_columns(column[:, None], tolerance)[:, 0]


def exact_heat_columns(
    laplacian: ArrayLike | sparse.spmatrix,
    diffusion_time: float,
    anchors: ArrayLike | None = None,
    *,
    batch_size: int = 128,
    tolerance: float = 1e-12,
) -> FloatArray:
    """Compute selected exact columns with batched sparse ``expm_multiply``.

    This is the scalable dense-reference backend used by kernel studies. It
    avoids dense ``scipy.linalg.expm`` and bounds temporary memory, but its
    returned array is still dense and must never be used by a method advertised
    as local.
    """

    if diffusion_time < 0 or not math.isfinite(diffusion_time):
        raise ValueError("diffusion_time must be finite and nonnegative")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    matrix = _as_square_laplacian(laplacian)
    if anchors is None:
        anchor_array = np.arange(matrix.shape[0], dtype=np.int64)
    else:
        anchor_array = np.asarray(anchors, dtype=np.int64)
        if anchor_array.ndim != 1:
            raise ValueError("anchors must be a one-dimensional sequence")
    if np.any(anchor_array < 0) or np.any(anchor_array >= matrix.shape[0]):
        raise IndexError("anchor index outside the action graph")
    result = np.empty((matrix.shape[0], len(anchor_array)), dtype=np.float64)
    for start in range(0, len(anchor_array), int(batch_size)):
        stop = min(start + int(batch_size), len(anchor_array))
        batch_anchors = anchor_array[start:stop]
        basis = np.zeros((matrix.shape[0], len(batch_anchors)), dtype=np.float64)
        basis[batch_anchors, np.arange(len(batch_anchors))] = 1.0
        result[:, start:stop] = expm_multiply(-diffusion_time * matrix, basis)
    return _clean_probability_columns(result, tolerance)


def diffusion_distance_gibbs_kernel(
    laplacian: ArrayLike | sparse.spmatrix,
    diffusion_time: float,
    lambda_: float,
    *,
    batch_size: int = 128,
) -> tuple[FloatArray, float, FloatArray]:
    """Uniform-prior Gibbs columns from squared diffusion distance.

    This kernel is a different LOT target from direct heat.  It lives in the
    numerical-kernel module so planning code need not depend on an experiment
    runner.
    """

    if lambda_ <= 0.0 or not math.isfinite(lambda_):
        raise ValueError("lambda_ must be finite and positive")
    heat_twice = exact_heat_columns(
        laplacian, 2.0 * diffusion_time, batch_size=batch_size
    )
    diagonal = np.diag(heat_twice).copy()
    costs = -2.0 * heat_twice
    costs += diagonal[:, None]
    costs += diagonal[None, :]
    np.maximum(costs, 0.0, out=costs)
    cost_max = float(costs.max(initial=0.0))
    log_weights = -costs / lambda_
    log_weights -= np.max(log_weights, axis=0, keepdims=True)
    weights = np.exp(log_weights)
    weights /= weights.sum(axis=0, keepdims=True)
    return weights, cost_max, diagonal


def uniformized_random_walk(
    laplacian: ArrayLike | sparse.spmatrix,
    nu_u: float | None = None,
    *,
    tolerance: float = 1e-12,
) -> tuple[sparse.csr_matrix, float]:
    """Return ``P_G = I - L / nu_u`` and the resolved rate."""

    matrix = _as_square_laplacian(laplacian)
    degrees = np.asarray(matrix.diagonal(), dtype=np.float64)
    max_degree = float(degrees.max(initial=0.0))
    resolved_rate = (
        max_degree
        if nu_u is None and max_degree > 0.0
        else (1.0 if nu_u is None else float(nu_u))
    )
    if resolved_rate <= 0.0 or not math.isfinite(resolved_rate):
        raise ValueError("nu_u must be finite and positive")
    if resolved_rate + tolerance < max_degree:
        raise ValueError(f"nu_u={resolved_rate} is below d_max={max_degree}")
    walk = sparse.eye(matrix.shape[0], format="csr") - matrix / resolved_rate
    walk.eliminate_zeros()
    if walk.nnz and np.min(walk.data) < -tolerance:
        raise FloatingPointError("uniformized walk contains negative probabilities")
    walk.data[walk.data < 0.0] = 0.0
    walk.eliminate_zeros()
    return walk.tocsr(), resolved_rate


def poisson_head_weights(theta: float, radius: int) -> FloatArray:
    """Conditional probabilities ``Pr(N=k | N<=radius)`` for Poisson ``N``."""

    if theta < 0.0 or not math.isfinite(theta):
        raise ValueError("theta must be finite and nonnegative")
    if not isinstance(radius, (int, np.integer)) or radius < 0:
        raise ValueError("radius must be a nonnegative integer")
    if theta == 0.0:
        weights = np.zeros(radius + 1, dtype=np.float64)
        weights[0] = 1.0
        return weights
    orders = np.arange(radius + 1, dtype=np.float64)
    log_weights = orders * math.log(theta) - gammaln(orders + 1.0)
    return np.exp(log_weights - logsumexp(log_weights))


def poisson_tail(theta: float, radius: int) -> float:
    if (
        theta < 0.0
        or not math.isfinite(theta)
        or not isinstance(radius, (int, np.integer))
        or radius < 0
    ):
        raise ValueError(
            "theta must be finite and nonnegative; radius must be a nonnegative integer"
        )
    return float(poisson.sf(radius, theta))


def heat_approximation_certificate(
    exact: ArrayLike,
    truncated: ArrayLike,
    beta_r: float,
    *,
    tolerance: float = 1e-12,
) -> HeatApproximationCertificate:
    """Check ``||H_t(:,j)-H_t,r(:,j)||_1 <= 2 beta_r`` by column.

    Vector inputs are interpreted as one column. Matrix inputs follow the
    repository convention: candidate actions index rows and anchors index
    columns.
    """

    exact_array = np.asarray(exact, dtype=np.float64)
    truncated_array = np.asarray(truncated, dtype=np.float64)
    if exact_array.shape != truncated_array.shape or exact_array.ndim not in {1, 2}:
        raise ValueError(
            "exact and truncated kernels must have matching vector or matrix shapes"
        )
    if not np.all(np.isfinite(exact_array)) or not np.all(
        np.isfinite(truncated_array)
    ):
        raise ValueError("kernel entries must be finite")
    if not 0.0 <= beta_r <= 1.0 or not math.isfinite(beta_r):
        raise ValueError("beta_r must lie in [0, 1]")
    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and nonnegative")
    if exact_array.ndim == 1:
        errors = np.array([np.abs(exact_array - truncated_array).sum()])
    else:
        errors = np.abs(exact_array - truncated_array).sum(axis=0)
    bound = 2.0 * beta_r
    maximum = float(errors.max(initial=0.0))
    return HeatApproximationCertificate(
        column_l1_errors=errors,
        max_column_l1_error=maximum,
        beta_r=float(beta_r),
        l1_bound=bound,
        tolerance=float(tolerance),
        holds=maximum <= bound + tolerance,
    )


def truncated_heat_kernel(
    laplacian: ArrayLike | sparse.spmatrix,
    diffusion_time: float,
    radius: int,
    *,
    nu_u: float | None = None,
) -> FloatArray:
    """Dense reference for the normalized Poisson head in Equation (32)."""

    if diffusion_time < 0.0 or not math.isfinite(diffusion_time):
        raise ValueError("diffusion_time must be finite and nonnegative")
    matrix = _as_square_laplacian(laplacian)
    walk, resolved_rate = uniformized_random_walk(matrix, nu_u)
    coefficients = poisson_head_weights(resolved_rate * diffusion_time, radius)
    power = np.eye(matrix.shape[0], dtype=np.float64)
    truncated = coefficients[0] * power
    for order in range(1, radius + 1):
        power = np.asarray(walk @ power)
        truncated += coefficients[order] * power
    return _clean_probability_columns(truncated)


def truncated_heat_column(
    laplacian: ArrayLike | sparse.spmatrix,
    diffusion_time: float,
    radius: int,
    anchor: int,
    *,
    nu_u: float | None = None,
) -> FloatArray:
    """Sparse-matvec reference for one normalized finite-walk column."""

    matrix = _as_square_laplacian(laplacian)
    if not 0 <= anchor < matrix.shape[0]:
        raise IndexError(anchor)
    walk, resolved_rate = uniformized_random_walk(matrix, nu_u)
    coefficients = poisson_head_weights(resolved_rate * diffusion_time, radius)
    power = np.zeros(matrix.shape[0], dtype=np.float64)
    power[anchor] = 1.0
    result = coefficients[0] * power
    for order in range(1, radius + 1):
        power = np.asarray(walk @ power).ravel()
        result += coefficients[order] * power
    return _clean_probability_columns(result[:, None])[:, 0]
