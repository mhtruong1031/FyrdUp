"""Shared bird's-eye fire coloring: red-dominant so VLMs see fire as fire, not terrain."""

# Draw cells the sim still treats as burning (slightly below physics threshold so edges show).
FIRE_DRAW_THRESHOLD = 0.05


def fire_cell_bgr(val: float) -> tuple[int, int, int]:
    """
    Map fire intensity 0..1 to a BGR tuple for OpenCV drawing.

    Old mapping used g=(1-val)*80 with r=val*255, so weak fire (e.g. val=0.1) was
    greener than red. This keeps red/orange/yellow progression like real fire.
    """
    v = max(0.0, min(1.0, val))
    r = int(min(255, 95 + 160 * v))
    g = int(min(230, 18 + 42 * v + 75 * v * v))
    b = int(min(90, 12 + 35 * v * v * v))
    return (b, g, r)
