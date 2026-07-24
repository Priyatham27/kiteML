# KiteML Typography & Wordmark Specification (`brand/typography.md`)

KiteML uses a modern, high-legibility geometric sans-serif typeface for the primary wordmark and documentation headers.

---

## 1. Primary Typeface Stack

```css
font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
```

- **Primary Font**: **Inter** (Google Fonts)
- **Fallback**: System Modern Geometric Sans-Serif (`Segoe UI`, `Roboto`)

---

## 2. Wordmark Construction Rules

The wordmark consists of two distinct components: **"Kite"** and **"ML"**.

```text
    K i t e M L
    ├─────┤ ├──┤
    Heavy   Bold (Electric Blue)
```

| Wordmark Part | Weight | Color (Light Mode) | Color (Dark Mode) | Letter Spacing |
| :--- | :--- | :--- | :--- | :--- |
| **"Kite"** | `800 (ExtraBold)` | `#0B0F19` (Deep Navy) | `#FFFFFF` (Pure White) | `-1.5px` (Tight optical tracking) |
| **"ML"** | `700 (Bold)` | `#0052FF` (Electric Blue) | `#0052FF` (Electric Blue) | `-1.5px` |

---

## 3. Optical Kerning & Spacing Guidelines

1. **No Space between Kite and ML**: The text string is typed as a single word `KiteML` without whitespace.
2. **Visual Hierarchy**: The color shift from `#0B0F19` to `#0052FF` provides clear visual separation while keeping the name unified.
3. **Alignment**: The vertical baseline of the wordmark is optically aligned with the vertical midpoint of the diamond kite icon.
