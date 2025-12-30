#!/usr/bin/env python3
"""Create a composite of remaining screenshots: syntax, compact-tty, numpy."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SCREENSHOTS_DIR = Path(__file__).parent.parent
OUTPUT_FILE = Path(__file__).parent.parent / "features-composite.webp"

# Target width to match chain-comparison.webp
TARGET_WIDTH = 2884
BACKGROUND_COLOR = (0, 0, 0, 0)  # Transparent
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


def create_composite():
    """Create composite with numpy on left, syntax and compact-tty stacked on right."""
    # Load images
    syntax = Image.open(SCREENSHOTS_DIR / "syntax-ipython.webp")
    compact = Image.open(SCREENSHOTS_DIR / "compact-tty.webp")
    numpy_img = Image.open(SCREENSHOTS_DIR / "numpy-jupyter.webp")

    print(f"  syntax-ipython: {syntax.width}x{syntax.height}")
    print(f"  compact-tty: {compact.width}x{compact.height}")
    print(f"  numpy-jupyter: {numpy_img.width}x{numpy_img.height}")

    # Right column height (syntax + compact stacked)
    right_col_height = syntax.height + compact.height
    right_col_width = compact.width  # compact is wider

    # Expand syntax to match compact width, filling with color from top-right corner
    fill_color = syntax.getpixel((syntax.width - 1, 0))
    syntax_expanded = Image.new("RGB", (right_col_width, syntax.height), fill_color)
    syntax_expanded.paste(syntax, (0, 0))

    # Scale numpy to match right column height
    numpy_scale = right_col_height / numpy_img.height
    numpy_scaled = numpy_img.resize(
        (int(numpy_img.width * numpy_scale), right_col_height), Image.Resampling.LANCZOS
    )

    # Calculate gap between columns
    # TARGET_WIDTH = numpy_scaled.width + gap + right_col_width
    gap = TARGET_WIDTH - numpy_scaled.width - right_col_width

    canvas_height = right_col_height

    print(f"\n  Numpy scaled: {numpy_scaled.width}x{numpy_scaled.height}")
    print(f"  Right column: {right_col_width}x{right_col_height}")
    print(f"  Gap between columns: {gap}")
    print(f"  Canvas: {TARGET_WIDTH}x{canvas_height}")

    # Create canvas with transparent background
    canvas = Image.new("RGBA", (TARGET_WIDTH, canvas_height), BACKGROUND_COLOR)

    # Place numpy on left
    canvas.paste(numpy_scaled, (0, 0))

    # Right column starts at this x position
    right_col_x = numpy_scaled.width + gap

    # Place syntax at top of right column, LEFT aligned (same left edge as compact)
    canvas.paste(syntax_expanded, (right_col_x, 0))

    # Place compact below syntax, left aligned
    canvas.paste(compact, (right_col_x, syntax.height))

    # Add labels
    font = get_font(72)

    # Add "Jupyter" label to numpy (top right of numpy area)
    draw = ImageDraw.Draw(canvas)

    # Jupyter label on numpy
    label = "Jupyter"
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    padding = 15
    x = numpy_scaled.width - text_width - padding * 2
    y = padding
    draw.rectangle(
        [
            x - padding,
            y - padding // 2,
            x + text_width + padding,
            y + text_height + padding,
        ],
        fill=LABEL_BG_COLOR,
    )
    draw.text((x, y), label, font=font, fill=LABEL_COLOR)

    # Terminal label at top right corner (in the empty space above compact, right of syntax)
    label = "Terminal"
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = TARGET_WIDTH - text_width - padding * 2
    y = padding
    draw.rectangle(
        [
            x - padding,
            y - padding // 2,
            x + text_width + padding,
            y + text_height + padding,
        ],
        fill=LABEL_BG_COLOR,
    )
    draw.text((x, y), label, font=font, fill=LABEL_COLOR)

    # Save
    canvas.save(OUTPUT_FILE, "WEBP", quality=90)
    print(f"\nCreated {OUTPUT_FILE}")
    print(f"  Output: {canvas.width}x{canvas.height}")


if __name__ == "__main__":
    create_composite()
