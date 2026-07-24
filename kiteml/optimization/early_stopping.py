"""
early_stopping.py — EarlyStopping manager for halting optimization search early in KiteML.
"""


class EarlyStopping:
    """
    Monitors trial execution and determines when optimization search should terminate.
    """

    def __init__(
        self,
        max_trials: int = 10,
        time_limit: float = 300.0,
        patience: int = 5,
    ) -> None:
        self.max_trials = max_trials
        self.time_limit = time_limit
        self.patience = patience
        self.best_score: float | None = None
        self.no_improvement_count: int = 0

    def should_stop(self, trial_count: int, elapsed_time: float, current_score: float) -> bool:
        """
        Check if termination condition is met.

        Parameters
        ----------
        trial_count : int
            Completed trial count.
        elapsed_time : float
            Total optimization duration in seconds.
        current_score : float
            Latest trial score.

        Returns
        -------
        bool
            True if search should stop, False otherwise.
        """
        if trial_count >= self.max_trials:
            return True

        if elapsed_time >= self.time_limit:
            return True

        if self.best_score is None or current_score > self.best_score:
            self.best_score = current_score
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        return self.no_improvement_count >= self.patience
