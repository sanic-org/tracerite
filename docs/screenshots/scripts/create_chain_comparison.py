#!/usr/bin/env python3
"""Create a side-by-side comparison of chain images."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SCREENSHOTS_DIR = Path(__file__).parent.parent
OUTPUT_FILE = Path(__file__).parent.parent / "chain-comparison.webp"

# Gap between images
GAP = 0
BACKGROUND_COLOR = "#1e1e1e"
LABEL_COLOR = "#ffffff"
LABEL_BG_COLOR = "#000000"


def get_font(size: int = 32):
    """Get a font for labels, falling back to default if needed."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def add_label(
    img: Image.Image, label: str, font: ImageFont.FreeTypeFont
) -> Image.Image:
    """Add a label at the top right corner of the image."""
    img = img.copy()
    draw = ImageDraw.Draw(img)

    # Get text size
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Position at top right with padding
    padding = 15
    x = img.width - text_width - padding * 2
    y = padding

    # Draw background rectangle
    draw.rectangle(
        [
            x - padding,
            y - padding // 2,
            x + text_width + padding,
            y + text_height + padding,
        ],
        fill=LABEL_BG_COLOR,
    )

    # Draw text
    draw.text((x, y), label, font=font, fill=LABEL_COLOR)

    return img


def create_comparison():
    """Create side-by-side comparison with matching heights."""
    # Load images
    without = Image.open(SCREENSHOTS_DIR / "chain-without-tracerite.webp")
    with_tr = Image.open(SCREENSHOTS_DIR / "chain-jupyter.webp")

    # Scale chain-jupyter to match height of chain-without-tracerite
    target_height = without.height
    scale = target_height / with_tr.height
    new_width = int(with_tr.width * scale)
    with_tr_scaled = with_tr.resize(
        (new_width, target_height), Image.Resampling.LANCZOS
    )

    # Add labels
    font = get_font(72)
    with_tr_labeled = add_label(with_tr_scaled, "With TraceRite", font)
    without_labeled = add_label(without, "Without", font)

    # Calculate canvas size (with on left, without on right)
    total_width = with_tr_labeled.width + GAP + without_labeled.width
    total_height = target_height

    # Create canvas and paste images
    canvas = Image.new("RGB", (total_width, total_height), BACKGROUND_COLOR)
    canvas.paste(with_tr_labeled, (0, 0))
    canvas.paste(without_labeled, (with_tr_labeled.width + GAP, 0))

    # Save
    canvas.save(OUTPUT_FILE, "WEBP", quality=90)
    print(f"Created {OUTPUT_FILE}")
    print(
        f"  Left (With): {with_tr.width}x{with_tr.height} → {with_tr_scaled.width}x{with_tr_scaled.height}"
    )
    print(f"  Right (Without): {without.width}x{without.height}")
    print(f"  Output: {total_width}x{total_height}")


if __name__ == "__main__":
    create_comparison()
