#!/usr/bin/env python3
"""Create a side-by-side comparison of chain images.

Both source images are scaled by the same factor so the text/font size is
identical. The output is fixed to a crisp width suitable for HiDPI web layouts.
The taller (builtin) image determines the canvas height; the shorter
(TraceRite) image is centred vertically, leaving black bars above and below.
Labels are placed in the empty/black space rather than the top-right corner.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SCREENSHOTS_DIR = Path(__file__).parent.parent
OUTPUT_FILE = Path(__file__).parent.parent / "chain-comparison.webp"

# Fixed output width for all README composites. 2400 px looks sharp on HiDPI
# displays when shown in a typical content column without filling the viewport.
FIXED_WIDTH = 2400

# Gap between images
GAP = 0
BACKGROUND_COLOR = (0, 0, 0, 255)  # Opaque black
LABEL_BG_COLOR = "#ffaa00"  # Yellow-orange advertising color


def get_font(size: int = 72):
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


def draw_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    font: ImageFont.FreeTypeFont,
    *,
    padding: int = 15,
    bg: bool = True,
) -> None:
    """Draw an explosive label: orange background, white text, black outline."""
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Extra bottom padding so the label doesn't look top-heavy.
    top_pad = padding // 2
    bottom_pad = int(padding * 1.8)

    if bg:
        draw.rectangle(
            [
                x - padding,
                y - top_pad,
                x + text_width + padding,
                y + text_height + bottom_pad,
            ],
            fill=LABEL_BG_COLOR,
        )

    # White text with a thick black outline for a three-color advertising look.
    draw.text(
        (x, y),
        text,
        font=font,
        fill="#ffffff",
        stroke_width=3,
        stroke_fill="#000000",
    )


def create_comparison():
    """Create side-by-side comparison with identical scaling."""
    with_img = Image.open(SCREENSHOTS_DIR / "chain-terminal.webp")
    without_img = Image.open(SCREENSHOTS_DIR / "chain-without-tracerite.webp")

    # Scale both by the same factor so the combined width matches FIXED_WIDTH.
    total_source_width = with_img.width + without_img.width
    scale = FIXED_WIDTH / total_source_width

    with_scaled = with_img.resize(
        (round(with_img.width * scale), round(with_img.height * scale)),
        Image.Resampling.LANCZOS,
    )
    without_scaled = without_img.resize(
        (round(without_img.width * scale), round(without_img.height * scale)),
        Image.Resampling.LANCZOS,
    )

    # Canvas height is driven by the taller builtin image.
    total_width = with_scaled.width + GAP + without_scaled.width
    total_height = max(with_scaled.height, without_scaled.height)

    canvas = Image.new("RGBA", (total_width, total_height), BACKGROUND_COLOR)

    # Builtin image fills the right side from the top.
    canvas.paste(without_scaled, (with_scaled.width + GAP, 0))

    draw = ImageDraw.Draw(canvas)
    font = get_font(40)
    padding = 10
    label_gap = 22  # extra breathing room below the top label

    # "TraceRite in terminal" in the top-left corner, above the TraceRite output.
    with_label = "TraceRite in terminal"
    with_bbox = draw.textbbox((0, 0), with_label, font=font)
    with_text_height = with_bbox[3] - with_bbox[1]
    with_label_x = padding
    with_label_y = 0
    draw_label(draw, with_label, with_label_x, with_label_y, font, bg=True)

    # Place the TraceRite image a little lower to avoid crowding the label.
    with_img_y = with_label_y + with_text_height + label_gap
    canvas.paste(with_scaled, (0, with_img_y))

    # "Without TraceRite" aligned so its background right edge sits on the middle seam.
    without_label = "Without TraceRite"
    without_bbox = draw.textbbox((0, 0), without_label, font=font)
    without_text_width = without_bbox[2] - without_bbox[0]
    without_text_height = without_bbox[3] - without_bbox[1]
    box_height = without_text_height + (padding // 2) + int(padding * 1.8)
    without_x = with_scaled.width + GAP - without_text_width - padding - 25
    without_y = total_height - without_text_height - 180 - without_text_height + box_height + box_height
    draw_label(draw, without_label, without_x, without_y, font, bg=True)

    canvas.save(OUTPUT_FILE, "WEBP", quality=90)
    print(f"Created {OUTPUT_FILE}")
    print(f"  With: {with_scaled.width}x{with_scaled.height} @ y={with_img_y}")
    print(f"  Without: {without_scaled.width}x{without_scaled.height}")
    print(f"  Output: {canvas.width}x{canvas.height}")


if __name__ == "__main__":
    create_comparison()
