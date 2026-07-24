#!/usr/bin/env python3
"""
generate_brand_assets.py — KiteML Brand Assets Generator

Generates production-grade SVG vector files and high-resolution PNG assets for KiteML:
- SVG: Primary logo, Dark logo, Light logo, Standalone icon, Favicon, Monochrome versions
- PNG: 1024x1024 App Icon, 1200x630 OpenGraph image, GitHub banner, PyPI banner, Documentation hero, Clearspace guide
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BRAND_DIR = Path("brand")
LOGO_DIR = BRAND_DIR / "logo"
ASSETS_DIR = BRAND_DIR / "assets"

LOGO_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# ── Color Constants ──────────────────────────────────────────────────
BLUE = "#0052FF"
NAVY = "#0B0F19"
WHITE = "#FFFFFF"
LIGHT_BG = "#F8FAFC"
DARK_BG = "#0B0F19"
GRAY_TEXT = "#64748B"
BORDER_GRAY = "#E2E8F0"

# ── SVG Vector Generators ───────────────────────────────────────────

def get_kiteml_icon_svg(width=200, height=200, monochrome=None, background=None):
    """Generates the geometric kite & circuit icon SVG string."""
    top_color = BLUE
    bottom_color = NAVY
    line_color = WHITE
    node_fill = WHITE
    node_inner = BLUE

    if monochrome == "black":
        top_color = "#000000"
        bottom_color = "#000000"
        line_color = WHITE
        node_fill = WHITE
        node_inner = "#000000"
    elif monochrome == "white":
        top_color = WHITE
        bottom_color = WHITE
        line_color = NAVY
        node_fill = NAVY
        node_inner = WHITE

    bg_rect = f'<rect width="{width}" height="{height}" fill="{background}"/>' if background else ""

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="{width}" height="{height}">
  {bg_rect}
  <g id="kite-geometry">
    <!-- Top Diamond Half -->
    <polygon points="100,12 182,100 100,100 18,100" fill="{top_color}"/>
    <!-- Bottom Diamond Half -->
    <polygon points="18,100 100,100 182,100 100,188" fill="{bottom_color}"/>
  </g>
  <g id="circuit-tree" stroke="{line_color}" stroke-linecap="round" stroke-linejoin="round">
    <!-- Central Vertical Trunk -->
    <line x1="100" y1="165" x2="100" y2="48" stroke-width="7"/>
    <!-- Left Angled Branch -->
    <line x1="100" y1="115" x2="62" y2="77" stroke-width="6"/>
    <!-- Right Angled Branch -->
    <line x1="100" y1="115" x2="138" y2="77" stroke-width="6"/>
    <!-- Top Center Branch -->
    <line x1="100" y1="70" x2="100" y2="48" stroke-width="6"/>
  </g>
  <g id="circuit-nodes">
    <!-- Top Node -->
    <circle cx="100" cy="48" r="9" fill="{node_fill}"/>
    <circle cx="100" cy="48" r="4.5" fill="{node_inner}"/>
    <!-- Left Node -->
    <circle cx="62" cy="77" r="8.5" fill="{node_fill}"/>
    <circle cx="62" cy="77" r="4" fill="{node_inner}"/>
    <!-- Right Node -->
    <circle cx="138" cy="77" r="8.5" fill="{node_fill}"/>
    <circle cx="138" cy="77" r="4" fill="{node_inner}"/>
  </g>
</svg>"""
    return svg


def get_kiteml_logo_svg(theme="light"):
    """Generates the primary horizontal logo (Icon + Wordmark)."""
    text_color_kite = NAVY if theme == "light" else WHITE
    text_color_ml = BLUE
    bg_color = LIGHT_BG if theme == "light" else DARK_BG

    top_color = BLUE
    bottom_color = NAVY if theme == "light" else "#1E293B"

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 160" width="600" height="160">
  <rect width="600" height="160" fill="{bg_color}" rx="12"/>
  <g transform="translate(30, 10)">
    <!-- Icon (scale 0.7) -->
    <g transform="translate(0, 0) scale(0.7)">
      <polygon points="100,12 182,100 100,100 18,100" fill="{top_color}"/>
      <polygon points="18,100 100,100 182,100 100,188" fill="{bottom_color}"/>
      <line x1="100" y1="165" x2="100" y2="48" stroke="{WHITE}" stroke-width="7" stroke-linecap="round"/>
      <line x1="100" y1="115" x2="62" y2="77" stroke="{WHITE}" stroke-width="6" stroke-linecap="round"/>
      <line x1="100" y1="115" x2="138" y2="77" stroke="{WHITE}" stroke-width="6" stroke-linecap="round"/>
      <circle cx="100" cy="48" r="9" fill="{WHITE}"/>
      <circle cx="100" cy="48" r="4.5" fill="{BLUE}"/>
      <circle cx="62" cy="77" r="8.5" fill="{WHITE}"/>
      <circle cx="62" cy="77" r="4" fill="{BLUE}"/>
      <circle cx="138" cy="77" r="8.5" fill="{WHITE}"/>
      <circle cx="138" cy="77" r="4" fill="{BLUE}"/>
    </g>
    <!-- Wordmark -->
    <text x="165" y="98" font-family="-apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif" font-size="64" font-weight="800" letter-spacing="-1.5">
      <tspan fill="{text_color_kite}">Kite</tspan>
      <tspan fill="{text_color_ml}">ML</tspan>
    </text>
  </g>
</svg>"""
    return svg


# ── PIL High-Res PNG Generators ──────────────────────────────────────

def draw_kiteml_icon_pil(draw, offset_x, offset_y, scale=1.0, dark_mode=False):
    """Draws the geometric kite & circuit icon onto a PIL ImageDraw context."""
    def s(val):
        return int(val * scale)

    ox, oy = offset_x, offset_y
    top_color = (0, 82, 255)
    bottom_color = (15, 23, 42) if dark_mode else (11, 15, 25)
    white = (255, 255, 255)
    blue = (0, 82, 255)

    # Top Polygon
    draw.polygon([
        (ox + s(100), oy + s(12)),
        (ox + s(182), oy + s(100)),
        (ox + s(100), oy + s(100)),
        (ox + s(18), oy + s(100))
    ], fill=top_color)

    # Bottom Polygon
    draw.polygon([
        (ox + s(18), oy + s(100)),
        (ox + s(100), oy + s(100)),
        (ox + s(182), oy + s(100)),
        (ox + s(100), oy + s(188))
    ], fill=bottom_color)

    # Lines
    draw.line([(ox + s(100), oy + s(165)), (ox + s(100), oy + s(48))], fill=white, width=max(1, s(7)))
    draw.line([(ox + s(100), oy + s(115)), (ox + s(62), oy + s(77))], fill=white, width=max(1, s(6)))
    draw.line([(ox + s(100), oy + s(115)), (ox + s(138), oy + s(77))], fill=white, width=max(1, s(6)))

    # Nodes
    r1, r1_in = max(2, s(9)), max(1, s(4.5))
    r2, r2_in = max(2, s(8.5)), max(1, s(4))

    # Top Node
    cx, cy = ox + s(100), oy + s(48)
    draw.ellipse([cx - r1, cy - r1, cx + r1, cy + r1], fill=white)
    draw.ellipse([cx - r1_in, cy - r1_in, cx + r1_in, cy + r1_in], fill=blue)

    # Left Node
    cx, cy = ox + s(62), oy + s(77)
    draw.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], fill=white)
    draw.ellipse([cx - r2_in, cy - r2_in, cx + r2_in, cy + r2_in], fill=blue)

    # Right Node
    cx, cy = ox + s(138), oy + s(77)
    draw.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], fill=white)
    draw.ellipse([cx - r2_in, cy - r2_in, cx + r2_in, cy + r2_in], fill=blue)


def create_app_icon_png():
    """Generates 1024x1024 high-res app icon / avatar PNG."""
    img = Image.new("RGBA", (1024, 1024), (11, 15, 25, 255))
    draw = ImageDraw.Draw(img)
    draw_kiteml_icon_pil(draw, offset_x=112, offset_y=112, scale=4.0, dark_mode=True)
    img.save(LOGO_DIR / "app-icon-1024.png", "PNG")


def create_social_preview_png():
    """Generates 1200x630 OpenGraph / Social preview banner."""
    img = Image.new("RGBA", (1200, 630), (11, 15, 25, 255))
    draw = ImageDraw.Draw(img)

    # Draw Large Icon
    draw_kiteml_icon_pil(draw, offset_x=100, offset_y=165, scale=1.5, dark_mode=True)

    # Draw Title Text
    try:
        font_large = ImageFont.truetype("arial.ttf", 96)
        font_sub = ImageFont.truetype("arial.ttf", 36)
        font_tag = ImageFont.truetype("arial.ttf", 28)
    except IOError:
        font_large = font_sub = font_tag = ImageFont.load_default()

    draw.text((440, 210), "Kite", fill=(255, 255, 255), font=font_large)
    draw.text((645, 210), "ML", fill=(0, 82, 255), font=font_large)
    draw.text((440, 330), "Intelligent Machine Learning Ecosystem", fill=(148, 163, 184), font=font_sub)
    draw.text((440, 390), "Production AutoML • DAG Pipelines • REST Serving • Drift Monitoring", fill=(0, 82, 255), font=font_tag)

    img.save(LOGO_DIR / "social-preview.png", "PNG")
    img.save(ASSETS_DIR / "og-image-1200x630.png", "PNG")


def create_github_banner_png():
    """Generates GitHub Repository Header Banner."""
    img = Image.new("RGBA", (1280, 400), (11, 15, 25, 255))
    draw = ImageDraw.Draw(img)

    draw_kiteml_icon_pil(draw, offset_x=120, offset_y=100, scale=1.0, dark_mode=True)

    try:
        font_large = ImageFont.truetype("arial.ttf", 80)
        font_sub = ImageFont.truetype("arial.ttf", 32)
    except IOError:
        font_large = font_sub = ImageFont.load_default()

    draw.text((360, 130), "Kite", fill=(255, 255, 255), font=font_large)
    draw.text((530, 130), "ML", fill=(0, 82, 255), font=font_large)
    draw.text((360, 230), "Train production-grade ML models with a single line of code.", fill=(148, 163, 184), font=font_sub)

    img.save(ASSETS_DIR / "github-banner.png", "PNG")


def create_pypi_banner_png():
    """Generates PyPI package header banner."""
    img = Image.new("RGBA", (1000, 300), (248, 250, 252, 255))
    draw = ImageDraw.Draw(img)

    draw_kiteml_icon_pil(draw, offset_x=80, offset_y=50, scale=1.0, dark_mode=False)

    try:
        font_large = ImageFont.truetype("arial.ttf", 72)
        font_sub = ImageFont.truetype("arial.ttf", 26)
    except IOError:
        font_large = font_sub = ImageFont.load_default()

    draw.text((310, 90), "Kite", fill=(11, 15, 25), font=font_large)
    draw.text((460, 90), "ML", fill=(0, 82, 255), font=font_large)
    draw.text((310, 180), "pip install kiteml-ai", fill=(0, 82, 255), font=font_sub)

    img.save(ASSETS_DIR / "pypi-banner.png", "PNG")


def create_documentation_hero_png():
    """Generates MkDocs documentation hero banner."""
    img = Image.new("RGBA", (1200, 400), (11, 15, 25, 255))
    draw = ImageDraw.Draw(img)

    draw_kiteml_icon_pil(draw, offset_x=100, offset_y=100, scale=1.0, dark_mode=True)

    try:
        font_large = ImageFont.truetype("arial.ttf", 72)
        font_sub = ImageFont.truetype("arial.ttf", 28)
    except IOError:
        font_large = font_sub = ImageFont.load_default()

    draw.text((340, 120), "KiteML Documentation", fill=(255, 255, 255), font=font_large)
    draw.text((340, 210), "Full-Stack Intelligent AutoML Architecture & API Reference", fill=(148, 163, 184), font=font_sub)

    img.save(ASSETS_DIR / "documentation-hero.png", "PNG")


def create_logo_clearspace_png():
    """Generates clearspace & alignment guide image."""
    img = Image.new("RGBA", (800, 400), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Grid / Clearspace box
    draw.rectangle([100, 80, 700, 320], outline=(226, 232, 240), width=2)
    draw_kiteml_icon_pil(draw, offset_x=150, offset_y=100, scale=1.0, dark_mode=False)

    try:
        font_large = ImageFont.truetype("arial.ttf", 64)
        font_label = ImageFont.truetype("arial.ttf", 18)
    except IOError:
        font_large = font_label = ImageFont.load_default()

    draw.text((380, 150), "Kite", fill=(11, 15, 25), font=font_large)
    draw.text((520, 150), "ML", fill=(0, 82, 255), font=font_large)

    # Clearspace markers X
    draw.text((110, 50), "Clearspace = 1X (Icon Height)", fill=(100, 116, 139), font=font_label)

    img.save(LOGO_DIR / "logo-clearspace.png", "PNG")


def main():
    print("Generating KiteML Brand Vector SVGs...")
    
    # Write SVG files
    with open(LOGO_DIR / "kiteml-logo.svg", "w", encoding="utf-8") as f:
        f.write(get_kiteml_logo_svg("light"))

    with open(LOGO_DIR / "kiteml-logo-light.svg", "w", encoding="utf-8") as f:
        f.write(get_kiteml_logo_svg("light"))

    with open(LOGO_DIR / "kiteml-logo-dark.svg", "w", encoding="utf-8") as f:
        f.write(get_kiteml_logo_svg("dark"))

    with open(LOGO_DIR / "kiteml-icon.svg", "w", encoding="utf-8") as f:
        f.write(get_kiteml_icon_svg(200, 200))

    with open(LOGO_DIR / "favicon.svg", "w", encoding="utf-8") as f:
        f.write(get_kiteml_icon_svg(32, 32))

    with open(LOGO_DIR / "kiteml-icon-black.svg", "w", encoding="utf-8") as f:
        f.write(get_kiteml_icon_svg(200, 200, monochrome="black"))

    with open(LOGO_DIR / "kiteml-icon-white.svg", "w", encoding="utf-8") as f:
        f.write(get_kiteml_icon_svg(200, 200, monochrome="white", background=NAVY))

    print("Generating High-Resolution PNG Brand Assets...")
    create_app_icon_png()
    create_social_preview_png()
    create_github_banner_png()
    create_pypi_banner_png()
    create_documentation_hero_png()
    create_logo_clearspace_png()

    print("All brand assets successfully generated in brand/!")


if __name__ == "__main__":
    main()
