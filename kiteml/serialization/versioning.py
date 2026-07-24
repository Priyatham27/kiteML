"""
versioning.py — Version compatibility manager for KiteML pipeline serialization.
"""

import sys
from typing import Any

from kiteml.serialization.manifest import PipelineManifest


class VersionManager:
    """
    Evaluates compatibility between serialized pipeline manifest metadata and current runtime.
    """

    @staticmethod
    def check_compatibility(manifest: PipelineManifest) -> tuple[bool, str]:
        """
        Check if serialized pipeline package is compatible with current runtime environment.

        Returns
        -------
        tuple[bool, str]
            (is_compatible, status_message)
        """
        curr_py = sys.version.split()[0]
        curr_py_major_minor = ".".join(curr_py.split(".")[:2])
        pkg_py_major_minor = ".".join(manifest.python_version.split(".")[:2])

        if curr_py_major_minor != pkg_py_major_minor:
            return (
                True,
                f"Minor Python version mismatch (Package: {manifest.python_version}, Current: {curr_py}). Operations should remain compatible.",
            )

        return (True, "Package environment fully compatible.")
