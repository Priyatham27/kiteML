"""
ui/colors.py — Terminal UI utilities for KiteML CLI.

Provides beautiful ANSI colors, banners, and status messages without heavy dependencies.
"""

import sys

from kiteml.utils.encoding import safe_print

# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"

_USE_COLORS = sys.stdout.isatty()


def set_colors(enabled: bool):
    global _USE_COLORS
    _USE_COLORS = enabled


def colorize(text: str, color: str) -> str:
    if not _USE_COLORS:
        return text
    return f"{color}{text}{RESET}"


def print_banner():
    """Print the KiteML CLI banner."""
    banner = f"""{BOLD}{BLUE}
 \U0001fa81 KiteML{RESET}{BLUE} \u2014 Machine Learning Framework
{RESET}"""
    safe_print(banner)


def print_step(message: str):
    icon = "\u2713"
    safe_print(f" {colorize(icon, GREEN)} {message}")


def print_info(message: str):
    icon = "\u2139"
    safe_print(f" {colorize(icon, BLUE)} {message}")


def print_warning(message: str):
    icon = "\u26a0"
    safe_print(f" {colorize(icon, YELLOW)} {message}")


def print_error(message: str):
    icon = "\u2716"
    safe_print(f" {colorize(icon, RED)} {message}")


def print_header(message: str):
    msg = "\u2500\u2500 " + message + " \u2500\u2500"
    safe_print(f"\n{BOLD}{colorize(msg, CYAN)}{RESET}")


def print_success(message: str):
    icon = "\U0001f389 "
    safe_print(f"\n{BOLD}{colorize(icon + message, GREEN)}{RESET}")

