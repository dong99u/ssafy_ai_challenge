from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "assets" / "cover-hero-v1.png"

SCALE = 2
W, H = 1600 * SCALE, 1000 * SCALE


def sc(v: int) -> int:
    return int(v * SCALE)


def rounded_rect(draw, box, radius, fill=None, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def checker_flag(draw, x, y, size, angle=0):
    cols = [(22, 50, 67, 255), (255, 255, 255, 240)]
    cell = size // 6
    for r in range(4):
      for c in range(5):
        fill = cols[(r + c) % 2]
        draw.rectangle(
            (x + c * cell, y + r * cell, x + (c + 1) * cell, y + (r + 1) * cell),
            fill=fill,
        )
    draw.line((x - sc(10), y + sc(5), x - sc(10), y + sc(110)), fill=(22, 50, 67, 255), width=sc(10))


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow)
    gdraw.ellipse((sc(760), sc(160), sc(1390), sc(790)), fill=(14, 116, 144, 36))
    gdraw.ellipse((sc(980), sc(210), sc(1570), sc(770)), fill=(144, 169, 85, 42))
    gdraw.ellipse((sc(620), sc(380), sc(1100), sc(920)), fill=(231, 154, 59, 26))
    glow = glow.filter(ImageFilter.GaussianBlur(sc(36)))
    img.alpha_composite(glow)

    panel = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(panel)
    rounded_rect(
        pdraw,
        (sc(520), sc(110), sc(1510), sc(870)),
        sc(46),
        fill=(255, 255, 255, 215),
        outline=(216, 222, 217, 255),
        width=sc(6),
    )
    panel = panel.filter(ImageFilter.GaussianBlur(sc(1)))
    img.alpha_composite(panel)

    draw = ImageDraw.Draw(img)

    # orbital ring
    draw.arc((sc(820), sc(250), sc(1230), sc(660)), start=198, end=332, fill=(14, 116, 144, 255), width=sc(22))
    draw.arc((sc(820), sc(250), sc(1230), sc(660)), start=336, end=44, fill=(144, 169, 85, 255), width=sc(22))
    draw.arc((sc(820), sc(250), sc(1230), sc(660)), start=92, end=152, fill=(231, 154, 59, 255), width=sc(22))

    # arrow heads
    draw.polygon(
        [(sc(1215), sc(300)), (sc(1295), sc(332)), (sc(1238), sc(396))],
        fill=(14, 116, 144, 255),
    )
    draw.polygon(
        [(sc(852), sc(548)), (sc(786), sc(640)), (sc(896), sc(674))],
        fill=(144, 169, 85, 255),
    )

    # central badge
    rounded_rect(
        draw,
        (sc(935), sc(318), sc(1105), sc(488)),
        sc(28),
        fill=(22, 50, 67, 255),
    )
    draw.ellipse((sc(985), sc(368), sc(1055), sc(438)), fill=(245, 245, 240, 255))
    rounded_rect(
        draw,
        (sc(995), sc(498), sc(1045), sc(530)),
        sc(12),
        fill=(22, 50, 67, 255),
    )

    # starting blocks / team runners
    rounded_rect(draw, (sc(700), sc(560), sc(805), sc(705)), sc(26), fill=(14, 116, 144, 235))
    rounded_rect(draw, (sc(824), sc(548), sc(954), sc(722)), sc(34), fill=(144, 169, 85, 235))
    rounded_rect(draw, (sc(972), sc(584), sc(1016), sc(680)), sc(20), fill=(231, 154, 59, 235))
    rounded_rect(draw, (sc(661), sc(537), sc(1085), sc(734)), sc(36), fill=(217, 238, 242, 170))

    # motion streaks
    for offset, color in [
        (0, (14, 116, 144, 170)),
        (sc(24), (144, 169, 85, 170)),
        (sc(48), (231, 154, 59, 140)),
    ]:
        draw.rounded_rectangle(
            (sc(560), sc(438) + offset, sc(748), sc(452) + offset),
            radius=sc(8),
            fill=color,
        )

    # flag + spark
    checker_flag(draw, sc(1305), sc(232), sc(140))
    draw.line((sc(1364), sc(370), sc(1364), sc(430)), fill=(22, 50, 67, 255), width=sc(10))
    for cx, cy, size in [(sc(760), sc(226), sc(16)), (sc(1186), sc(204), sc(14)), (sc(1276), sc(548), sc(18))]:
        draw.line((cx - size, cy, cx + size, cy), fill=(231, 154, 59, 220), width=sc(4))
        draw.line((cx, cy - size, cx, cy + size), fill=(231, 154, 59, 220), width=sc(4))

    # podium stripe
    draw.rounded_rectangle(
        (sc(740), sc(744), sc(1365), sc(770)),
        radius=sc(12),
        fill=(22, 50, 67, 40),
    )

    final_img = img.resize((W // SCALE, H // SCALE), Image.Resampling.LANCZOS)
    final_img.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
