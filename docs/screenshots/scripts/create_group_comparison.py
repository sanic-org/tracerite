#!/usr/bin/env python3
"""Create a side-by-side comparison of ExceptionGroup in HTML and TTY."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SCREENSHOTS_DIR = Path(__file__).parent.parent
OUTPUT_FILE = Path(__file__).parent.parent / "group-comparison.webp"

# Target width to match chain-comparison.webp
TARGET_WIDTH = 2884
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
    """Create side-by-side comparison with matching heights, fitting target width."""
    # Load images
    html_img = Image.open(SCREENSHOTS_DIR / "group-html.webp")
    tty_img = Image.open(SCREENSHOTS_DIR / "group-tty.webp")

    print(f"  HTML: {html_img.width}x{html_img.height}")
    print(f"  TTY: {tty_img.width}x{tty_img.height}")

    # Calculate scale to fit target width while matching heights
    # Let h be the common height, then:
    # html_width_scaled = html_img.width * (h / html_img.height)
    # tty_width_scaled = tty_img.width * (h / tty_img.height)
    # html_width_scaled + tty_width_scaled = TARGET_WIDTH
    # h * (html_img.width/html_img.height + tty_img.width/tty_img.height) = TARGET_WIDTH

    html_ratio = html_img.width / html_img.height
    tty_ratio = tty_img.width / tty_img.height

    target_height = int(TARGET_WIDTH / (html_ratio + tty_ratio))

    # Scale both images to target height
    html_scaled_width = int(html_img.width * target_height / html_img.height)
    tty_scaled_width = int(tty_img.width * target_height / tty_img.height)

    html_scaled = html_img.resize(
        (html_scaled_width, target_height), Image.Resampling.LANCZOS
    )
    tty_scaled = tty_img.resize(
        (tty_scaled_width, target_height), Image.Resampling.LANCZOS
    )

    # Add labels
    font = get_font(72)
    html_labeled = add_label(html_scaled, "HTML", font)
    tty_labeled = add_label(tty_scaled, "Terminal", font)

    # Calculate canvas size (HTML on left, TTY on right)
    total_width = html_labeled.width + tty_labeled.width
    total_height = target_height

    # Create canvas and paste images
    canvas = Image.new("RGB", (total_width, total_height), BACKGROUND_COLOR)
    canvas.paste(html_labeled, (0, 0))
    canvas.paste(tty_labeled, (html_labeled.width, 0))

    # Save
    canvas.save(OUTPUT_FILE, "WEBP", quality=90)
    print(f"\nCreated {OUTPUT_FILE}")
    print(f"  HTML scaled: {html_scaled.width}x{html_scaled.height}")
    print(f"  TTY scaled: {tty_scaled.width}x{tty_scaled.height}")
    print(f"  Output: {total_width}x{total_height}")


if __name__ == "__main__":
    create_comparison()
