"""
config.py — KiteML Global Configuration.

All tunable defaults live here.  No module should hard-code 42, 0.2, or 5
directly — import from here instead.  This means changing a default is a
one-line edit in one file.

Usage
-----
    from kiteml.config import DEFAULT_RANDOM_STATE, DEFAULT_CV_FOLDS

Overriding per-run
------------------
Every public function that uses these values also accepts them as explicit
keyword arguments, so callers can override without touching this file:

    kiteml.train(df, target="y", random_state=0)
"""

# ── Reproducibility ───────────────────────────────────────────────────────────
DEFAULT_RANDOM_STATE: int = 42
"""Global random seed used by all estimators and train/test splits."""

# ── Data splitting ────────────────────────────────────────────────────────────
DEFAULT_TEST_SIZE: float = 0.2
"""Fraction of data held out for the final evaluation set."""

# ── Cross-validation ──────────────────────────────────────────────────────────
DEFAULT_CV_FOLDS: int = 5
"""Number of folds used by the model selector during cross-validation."""

# ── Parallelism ───────────────────────────────────────────────────────────────
DEFAULT_N_JOBS: int = -1
"""
n_jobs value forwarded to cross_val_score and tree-based estimators.
-1 means "use all available CPU cores".
"""

# ── Verbose ───────────────────────────────────────────────────────────────────
DEFAULT_VERBOSE: bool = True
"""Emit INFO-level progress messages during training when True."""
