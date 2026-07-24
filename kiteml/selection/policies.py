"""
policies.py — SelectionPolicy weights generator for KiteML model selection.
"""


class SelectionPolicy:
    """
    Defines policy weighting profiles for model selection.
    """

    SUPPORTED_POLICIES = {"balanced", "accuracy", "fast_inference", "low_memory"}

    def get_weights(self, policy: str = "balanced") -> dict[str, float]:
        """
        Get metric weighting profile.

        Parameters
        ----------
        policy : str
            Policy profile name.

        Returns
        -------
        dict[str, float]
            Dictionary of weights for performance, stability, speed, and memory.
        """
        p = policy.lower()
        if p == "accuracy":
            return {"performance": 0.85, "stability": 0.10, "speed": 0.03, "memory": 0.02}
        if p == "fast_inference":
            return {"performance": 0.30, "stability": 0.10, "speed": 0.50, "memory": 0.10}
        if p == "low_memory":
            return {"performance": 0.30, "stability": 0.10, "speed": 0.10, "memory": 0.50}

        # Default balanced policy
        return {"performance": 0.50, "stability": 0.20, "speed": 0.15, "memory": 0.15}
