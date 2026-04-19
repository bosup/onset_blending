# ==============================================================================
# File: misc.py
# ==============================================================================
# Purpose
#   Small utility functions used across multiple pipeline stages.
#
# Function index
#   coalesce(x, y)
#     Null-coalescing helper. Returns x if not None, else y.
#
#   softmax_row(x)
#     Numerically stable softmax for a single numeric array.
#
#   row_logsumexp(mat_NK)
#     Row-wise log-sum-exp for an N x K matrix.
#
#   inv_logit(x)
#     Numerically stable inverse logit (sigmoid).
# ==============================================================================

import numpy as np


def coalesce(x, y):
    """Return x if not None, else y. Equivalent to R's %||% operator."""
    return x if x is not None else y


def softmax_row(x):
    """Numerically stable softmax for a 1D numeric array."""
    x = np.asarray(x, dtype=float)
    z = x - np.nanmax(x)
    ez = np.exp(z)
    return ez / np.sum(ez)


def row_logsumexp(mat_NK):
    """Row-wise log-sum-exp for an N x K matrix."""
    mat = np.asarray(mat_NK, dtype=float)
    m = mat.max(axis=1)
    return m + np.log(np.sum(np.exp(mat - m[:, None]), axis=1))


def inv_logit(x):
    """Numerically stable inverse logit (sigmoid)."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (np.exp(x) + 1.0))
