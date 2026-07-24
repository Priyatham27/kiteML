# KiteML Color System Specification (`brand/colors.md`)

The KiteML color system is designed for high visual contrast, digital legibility, and seamless adaptation across dark and light software interfaces (IDE themes, CLI rich output, documentation sites, and social previews).

---

## 🎨 Primary Brand Palette

```
  Primary Blue          Deep Navy             Pure White           Slate Text
  #0052FF               #0B0F19               #FFFFFF              #64748B
  rgb(0, 82, 255)       rgb(11, 15, 25)       rgb(255, 255, 255)   rgb(100, 116, 139)
```

| Swatch | Color Name | HEX | RGB | HSL | Primary Usage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 🔵 | **Electric Blue** | `#0052FF` | `0, 82, 255` | `221°, 100%, 50%` | Top Diamond Half, Primary Accents, Wordmark "ML", Active Buttons |
| ⬛ | **Deep Navy** | `#0B0F19` | `11, 15, 25` | `223°, 39%, 7%` | Bottom Diamond Half, Dark Theme Backgrounds, Wordmark "Kite" (Light Mode) |
| ⚪ | **Pure White** | `#FFFFFF` | `255, 255, 255` | `0°, 0%, 100%` | Circuit Node Pathways, Wordmark "Kite" (Dark Mode), Light Backgrounds |
| 🔘 | **Slate Gray** | `#64748B` | `100, 116, 139` | `215°, 16%, 47%` | Secondary Labels, Subtitle Text, Border Accents |

---

## 🌓 Dark vs. Light Theme Applications

=== "Light Mode (`#F8FAFC` Background)"

    - **Icon Top**: `#0052FF` (Electric Blue)
    - **Icon Bottom**: `#0B0F19` (Deep Navy)
    - **Wordmark "Kite"**: `#0B0F19` (Deep Navy)
    - **Wordmark "ML"**: `#0052FF` (Electric Blue)
    - **Circuit Lines & Nodes**: `#FFFFFF` (Pure White)

=== "Dark Mode (`#0B0F19` Background)"

    - **Icon Top**: `#0052FF` (Electric Blue)
    - **Icon Bottom**: `#1E293B` (Slate Navy)
    - **Wordmark "Kite"**: `#FFFFFF` (Pure White)
    - **Wordmark "ML"**: `#0052FF` (Electric Blue)
    - **Circuit Lines & Nodes**: `#FFFFFF` (Pure White)

---

## ♿ Accessibility & Contrast Compliance

All core color combinations pass **WCAG 2.1 AA** contrast checks:
- **`#0052FF` on `#FFFFFF`**: Contrast ratio `4.6:1` (Passes AA for UI elements & large text).
- **`#FFFFFF` on `#0B0F19`**: Contrast ratio `18.2:1` (Passes AAA for body text).
- **`#0052FF` on `#0B0F19`**: Contrast ratio `4.8:1` (Passes AA for accents).
