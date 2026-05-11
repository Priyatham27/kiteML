"""
encoding.py — Cross-platform safe print utility.

Handles the UnicodeEncodeError that occurs on Windows when printing
emoji/Unicode characters to a terminal using cp1252 encoding.
"""

import sys


def safe_print(*args, **kwargs):
    """
    Print with graceful Unicode fallback.

    On Windows terminals that don't support UTF-8 (cp1252, etc.),
    emoji and special characters are replaced with '?' instead of
    raising UnicodeEncodeError.
    """
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Fallback: encode with replacement and decode back
        text = " ".join(str(a) for a in args)
        encoding = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
        safe_text = text.encode(encoding, errors="replace").decode(encoding)
        print(safe_text, **kwargs)
