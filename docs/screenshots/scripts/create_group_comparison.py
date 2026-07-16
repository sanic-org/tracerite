#!/usr/bin/env python3
"""Create a side-by-side comparison of ExceptionGroup in HTML and TTY.

Both source images are scaled by the same factor so the text/font size is
identical. The output is fixed to a crisp width suitable for HiDPI web layouts.
The taller (HTML) image determines the canvas height; the shorter
(TTY) image is placed lower, leaving black space above it for the Terminal
label.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SCREENSHOTS_DIR = Path(__file__).parent.parent
OUTPUT_FILE = Path(__file__).parent.parent / "group-comparison.webp"

# Fixed output width for all README composites. 2400 px looks sharp on HiDPI
# displays when shown in a typical content column without filling the viewport.
FIXED_WIDTH = 2400

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
    html_img = Image.open(SCREENSHOTS_DIR / "group-html.webp")
    tty_img = Image.open(SCREENSHOTS_DIR / "group-tty.webp")

    print(f"  HTML: {html_img.width}x{html_img.height}")
    print(f"  TTY: {tty_img.width}x{tty_img.height}")

    # Scale both by the same factor so the combined width matches FIXED_WIDTH.
    total_source_width = html_img.width + tty_img.width
    scale = FIXED_WIDTH / total_source_width

    html_scaled = html_img.resize(
        (round(html_img.width * scale), round(html_img.height * scale)),
        Image.Resampling.LANCZOS,
    )
    tty_scaled = tty_img.resize(
        (round(tty_img.width * scale), round(tty_img.height * scale)),
        Image.Resampling.LANCZOS,
    )

    # Canvas height is driven by the taller HTML image.
    total_width = html_scaled.width + GAP + tty_scaled.width
    total_height = max(html_scaled.height, tty_scaled.height)

    canvas = Image.new("RGBA", (total_width, total_height), BACKGROUND_COLOR)

    # HTML image fills the left side from the top.
    canvas.paste(html_scaled, (0, 0))

    draw = ImageDraw.Draw(canvas)
    font = get_font(40)
    padding = 10

    # "HTML" label near the middle line, but with extra margin from the seam.
    html_label = "HTML"
    html_bbox = draw.textbbox((0, 0), html_label, font=font)
    html_text_width = html_bbox[2] - html_bbox[0]
    html_x = html_scaled.width - html_text_width - 80
    html_y = padding
    draw_label(draw, html_label, html_x, html_y, font, bg=True)

    # "Terminal" label just to the right of the middle line, mirroring the HTML
    # label's 80 px margin from the seam.
    tty_label = "Terminal"
    tty_bbox = draw.textbbox((0, 0), tty_label, font=font)
    tty_bbox[2] - tty_bbox[0]
    tty_label_x = html_scaled.width + GAP + 80
    tty_label_y = padding
    draw_label(draw, tty_label, tty_label_x, tty_label_y, font, bg=True)

    # Place the terminal image at the bottom of the right half.
    tty_img_y = total_height - tty_scaled.height
    canvas.paste(tty_scaled, (html_scaled.width + GAP, tty_img_y))

    canvas.save(OUTPUT_FILE, "WEBP", quality=90)
    print(f"\nCreated {OUTPUT_FILE}")
    print(f"  HTML scaled: {html_scaled.width}x{html_scaled.height}")
    print(f"  TTY scaled: {tty_scaled.width}x{tty_scaled.height} @ y={tty_img_y}")
    print(f"  Output: {canvas.width}x{canvas.height}")


if __name__ == "__main__":
    create_comparison()
