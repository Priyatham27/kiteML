"""
ui/colors.py — Terminal UI utilities for KiteML CLI.

Provides beautiful ANSI colors, banners, and status messages without heavy dependencies.
"""

import sys

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
 🪁 KiteML{RESET}{BLUE} — Machine Learning Framework
{RESET}"""
    print(banner)


def print_step(message: str):
    print(f" {colorize('✓', GREEN)} {message}")


def print_info(message: str):
    print(f" {colorize('ℹ', BLUE)} {message}")


def print_warning(message: str):
    print(f" {colorize('⚠', YELLOW)} {message}")


def print_error(message: str):
    print(f" {colorize('✖', RED)} {message}")


def print_header(message: str):
    print(f"\n{BOLD}{colorize('── ' + message + ' ──', CYAN)}{RESET}")


def print_success(message: str):
    print(f"\n{BOLD}{colorize('🎉 ' + message, GREEN)}{RESET}")
