#!/usr/bin/env python3
"""Create a composite of remaining screenshots: syntax, compact-tty, numpy.

The output is fixed to the same crisp width used by the other README composites.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SCREENSHOTS_DIR = Path(__file__).parent.parent
OUTPUT_FILE = Path(__file__).parent.parent / "features-composite.webp"

# Fixed output width for all README composites. 2400 px looks sharp on HiDPI
# displays when shown in a typical content column without filling the viewport.
FIXED_WIDTH = 2400

BACKGROUND_COLOR = (0, 0, 0, 0)  # Transparent
LABEL_BG_COLOR = "#ffaa00"  # Yellow-orange advertising color


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


def draw_label_at_top_right(
    draw: ImageDraw.ImageDraw,
    text: str,
    right: int,
    top: int,
    font: ImageFont.FreeTypeFont,
    *,
    padding: int = 10,
) -> None:
    """Draw a label whose bounding box is flush with the image's top-right corner.

    The text is inset by ``padding`` inside the coloured background so it does
    not touch the edges, but the background rectangle itself has no gap from
    the corner.
    """
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Same asymmetric padding as the other composites, but the whole label is
    # anchored at the top-right corner rather than floating with a margin.
    top_pad = padding // 2
    bottom_pad = int(padding * 1.8)

    x = right - text_width - padding
    y = top + top_pad

    draw.rectangle(
        [
            x - padding,
            y - top_pad,
            x + text_width + padding,
            y + text_height + bottom_pad,
        ],
        fill=LABEL_BG_COLOR,
    )

    draw.text(
        (x, y),
        text,
        font=font,
        fill="#ffffff",
        stroke_width=3,
        stroke_fill="#000000",
    )


def create_composite():
    """Create composite with numpy on left, syntax and compact-tty stacked on right."""
    syntax = Image.open(SCREENSHOTS_DIR / "syntax-ipython.webp")
    compact = Image.open(SCREENSHOTS_DIR / "compact-tty.webp")
    numpy_img = Image.open(SCREENSHOTS_DIR / "numpy-jupyter.webp")

    print(f"  syntax-ipython: {syntax.width}x{syntax.height}")
    print(f"  compact-tty: {compact.width}x{compact.height}")
    print(f"  numpy-jupyter: {numpy_img.width}x{numpy_img.height}")

    # Natural layout: numpy scaled so its height matches the right column.
    right_col_height = syntax.height + compact.height
    right_col_width = compact.width  # compact is wider
    numpy_scale = right_col_height / numpy_img.height
    numpy_width = int(numpy_img.width * numpy_scale)

    natural_width = numpy_width + right_col_width
    fit_scale = FIXED_WIDTH / natural_width

    # Scale everything to the fixed output width.
    numpy_scaled = numpy_img.resize(
        (
            round(numpy_img.width * numpy_scale * fit_scale),
            round(numpy_img.height * numpy_scale * fit_scale),
        ),
        Image.Resampling.LANCZOS,
    )
    syntax_scaled = syntax.resize(
        (round(syntax.width * fit_scale), round(syntax.height * fit_scale)),
        Image.Resampling.LANCZOS,
    )
    compact_scaled = compact.resize(
        (round(compact.width * fit_scale), round(compact.height * fit_scale)),
        Image.Resampling.LANCZOS,
    )

    # Recompute right column after scaling.
    right_col_height = syntax_scaled.height + compact_scaled.height
    right_col_width = compact_scaled.width
    gap = 0

    canvas_width = numpy_scaled.width + gap + right_col_width
    canvas_height = right_col_height

    print(f"\n  Numpy scaled: {numpy_scaled.width}x{numpy_scaled.height}")
    print(f"  Syntax scaled: {syntax_scaled.width}x{syntax_scaled.height}")
    print(f"  Compact scaled: {compact_scaled.width}x{compact_scaled.height}")
    print(f"  Right column: {right_col_width}x{right_col_height}")
    print(f"  Canvas: {canvas_width}x{canvas_height}")

    canvas = Image.new("RGBA", (canvas_width, canvas_height), BACKGROUND_COLOR)

    # Place numpy on the left.
    canvas.paste(numpy_scaled, (0, 0))

    # Right column starts here.
    right_col_x = numpy_scaled.width + gap

    # Expand syntax to the full right-column width, filling from the top-right
    # corner so the prompt arrow stays aligned with compact below.
    fill_color = syntax_scaled.getpixel((syntax_scaled.width - 1, 0))
    syntax_expanded = Image.new(
        "RGB", (right_col_width, syntax_scaled.height), fill_color
    )
    syntax_expanded.paste(syntax_scaled, (0, 0))
    canvas.paste(syntax_expanded, (right_col_x, 0))

    # Place compact below syntax, left aligned.
    canvas.paste(compact_scaled, (right_col_x, syntax_scaled.height))

    # Labels: same size as the other composites so they scale consistently.
    font = get_font(40)
    draw = ImageDraw.Draw(canvas)

    # Notebook/HTML label at the top-right corner of the numpy image.
    draw_label_at_top_right(
        draw, "Notebook/HTML", right=numpy_scaled.width, top=0, font=font, padding=10
    )

    # Terminal label at the top-right corner of the whole composite.
    draw_label_at_top_right(
        draw,
        "Terminal",
        right=canvas_width,
        top=0,
        font=font,
        padding=10,
    )

    canvas.save(OUTPUT_FILE, "WEBP", quality=90)
    print(f"\nCreated {OUTPUT_FILE}")
    print(f"  Output: {canvas.width}x{canvas.height}")


if __name__ == "__main__":
    create_composite()
