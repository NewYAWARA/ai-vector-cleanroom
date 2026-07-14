"""Generate deterministic PNG fixtures for the black-box regression suite."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def _rgb(size=(512, 512), color="white"):
    image = Image.new("RGB", size, color)
    return image, ImageDraw.Draw(image)


def generate(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)

    image, draw = _rgb()
    draw.line((56, 256, 456, 256), fill="black", width=1)
    image.save(output / "one_px_black.png")

    image, draw = _rgb()
    draw.line((56, 256, 456, 256), fill=(221, 221, 221), width=12)
    image.save(output / "low_contrast_ddd.png")

    image, draw = _rgb()
    draw.line(((80, 90), (250, 90), (250, 400)),
              fill="black", width=14, joint="curve")
    image.save(output / "right_angle.png")

    image, draw = _rgb()
    draw.rectangle((90, 90, 422, 422), outline="black", width=5)
    image.save(output / "square_frame_5px.png")

    image, draw = _rgb()
    draw.ellipse((75, 75, 437, 437), outline="black", width=18)
    image.save(output / "ring.png")

    image, draw = _rgb()
    draw.line((80, 110, 432, 110), fill="black", width=12)
    draw.line((256, 110, 256, 430), fill="black", width=12)
    image.save(output / "t_junction.png")

    image, draw = _rgb()
    draw.line((90, 90, 422, 422), fill="black", width=12)
    draw.line((422, 90, 90, 422), fill="black", width=12)
    image.save(output / "x_junction.png")

    image, draw = _rgb()
    draw.line((100, 100, 256, 256), fill="black", width=12)
    draw.line((412, 100, 256, 256), fill="black", width=12)
    draw.line((256, 256, 256, 440), fill="black", width=12)
    image.save(output / "y_junction.png")

    image, draw = _rgb()
    draw.line((55, 256, 256, 256), fill=(220, 20, 30), width=14)
    draw.line((256, 256, 457, 256), fill=(20, 70, 220), width=14)
    image.save(output / "multicolor_touch.png")

    image, draw = _rgb()
    draw.rounded_rectangle((80, 120, 432, 390), radius=35,
                           fill=(245, 205, 20))
    draw.line((55, 300, 457, 190), fill=(51, 51, 51), width=8)
    image.save(output / "line_over_fill_darkgray.png")

    image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.line((56, 256, 456, 256), fill=(0, 0, 0, 100), width=12)
    image.save(output / "soft_alpha_100.png")

    image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((55, 146, 245, 336), fill=(225, 35, 45, 80))
    draw.ellipse((267, 146, 457, 336), fill=(25, 75, 225, 200))
    image.save(output / "mixed_alpha.png")

    image, draw = _rgb((3000, 3000))
    draw.line((100, 1500, 2900, 1500), fill="black", width=1)
    image.save(output / "one_px_black_3000.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "fixtures")
    args = parser.parse_args()
    generate(args.output.resolve())
    print(f"Generated 13 fixtures in {args.output.resolve()}")


if __name__ == "__main__":
    main()
