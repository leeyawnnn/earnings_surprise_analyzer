"""One visual grammar for every figure in the repo.

Charts are read at GitHub's rendered width, which is roughly 880 pixels, so
anything wider is downsampled and thin lines and small type disappear. The
canvas here is 1200x750 CSS pixels, vector where there is text and raster only
where a scatter is genuinely too dense to draw as paths.

The palette is Okabe-Ito, which stays distinguishable under the common forms
of colour blindness. Nothing in these figures is encoded by colour alone:
every series is either directly labelled or carries a distinct marker.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

#: Okabe & Ito (2008), "Color Universal Design". Eight hues that remain
#: separable under deuteranopia, protanopia and tritanopia.
OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}

#: Roles, so that a bucket keeps the same colour in every figure.
BEAT = OKABE_ITO["blue"]
MISS = OKABE_ITO["vermillion"]
INLINE = OKABE_ITO["orange"]
NEUTRAL = "#5A5A5A"
GRID = "#D8D8D8"
INK = "#1A1A1A"

CATEGORY_COLORS = {"Beat": BEAT, "Miss": MISS, "In-Line": INLINE}
CATEGORY_MARKERS = {"Beat": "o", "Miss": "s", "In-Line": "^"}

#: 1200x750 px at 100 dpi for vector output.
FIGSIZE = (12.0, 7.5)
#: Raster figures use the same canvas, so type and margins match the vector
#: ones, and are written at the highest density that still respects the
#: 1400-pixel ceiling: 12 inches at 116 dpi is 1392 px. Rendering them at a
#: nominal 200 dpi would give a 2400 px image that GitHub downsamples anyway,
#: which costs file size and buys nothing.
RASTER_FIGSIZE = FIGSIZE
RASTER_DPI = 116

#: DejaVu Sans ships with matplotlib, so it is the guaranteed fallback and is
#: what CI renders with. The earlier entries are used when installed locally.
FONT_STACK = ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans", "sans-serif"]

TITLE_SIZE = 15
SUBTITLE_SIZE = 11
LABEL_SIZE = 12
TICK_SIZE = 10
ANNOTATION_SIZE = 9
SOURCE_SIZE = 8


def use_house_style() -> None:
    """Apply the shared rcParams. Idempotent; call once per figure module."""
    mpl.rcParams.update(
        {
            "figure.figsize": FIGSIZE,
            "figure.dpi": 100,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": None,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "font.size": TICK_SIZE,
            "text.color": INK,
            "axes.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.labelsize": LABEL_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.9,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            # Keep SVG output byte-stable across runs.
            "svg.hashsalt": "earnings-surprise-analyzer",
            # Write text as text rather than as glyph outlines. Outlines
            # render identically everywhere but cost roughly ten times the
            # file size, which is why the first draft of these figures came to
            # two megabytes. The font stack below ends in a generic family, so
            # a viewer without the named fonts still gets a sans-serif face.
            "svg.fonttype": "none",
            # Drop vertices that would land within a third of a pixel of the
            # line they are on. A daily series of several thousand points is
            # indistinguishable from its simplified path at the width GitHub
            # renders these at, and the underlying data is unchanged.
            "path.simplify": True,
            "path.simplify_threshold": 0.4,
        }
    )


def horizontal_grid(ax: plt.Axes) -> None:
    """Light horizontal rules only, which is all a value axis needs."""
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.grid(False, axis="x")


def percent_axis(ax: plt.Axes, axis: str = "y", decimals: int = 1) -> None:
    """Format an axis as ``12.3%`` rather than ``0.123`` or a bare number."""
    formatter = FuncFormatter(lambda v, _: f"{v:.{decimals}f}%")
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_formatter(formatter)


#: Characters per subtitle line at 11pt on the standard canvas. Measured
#: rather than guessed: longer lines run off the right edge, which is how the
#: first draft of every one of these figures lost its last few words.
SUBTITLE_WRAP = 118


def titles(fig: Figure, title: str, subtitle: str) -> None:
    """Finding on top, sample and units underneath.

    The title says what the chart shows the reader; the subtitle carries the
    sample, the period and the units, so neither has to be guessed from the
    axes. The subtitle wraps rather than overflowing, and every figure leaves
    room for two lines of it so panels stay aligned across the set.
    """
    fig.suptitle(title, fontsize=TITLE_SIZE, fontweight="semibold", x=0.055, ha="left", y=0.972)
    wrapped = textwrap.fill(subtitle, SUBTITLE_WRAP)
    fig.text(0.055, 0.922, wrapped, fontsize=SUBTITLE_SIZE, color=NEUTRAL, ha="left", va="top")


#: Characters per source-note line at 8pt on the standard canvas.
SOURCE_WRAP = 155


def source_note(fig: Figure, text: str) -> None:
    """Provenance footer, bottom left, 8pt italic, wrapped to the canvas."""
    wrapped = textwrap.fill(text, SOURCE_WRAP)
    fig.text(
        0.055, 0.012, wrapped, fontsize=SOURCE_SIZE, style="italic", color=NEUTRAL, ha="left", va="bottom"
    )


def callout(
    ax: plt.Axes,
    text: str,
    xy: tuple[float, float],
    xytext: tuple[float, float],
    *,
    color: str = INK,
) -> None:
    """A short reader takeaway with a leader line."""
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        fontsize=ANNOTATION_SIZE,
        color=color,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.9, "shrinkA": 0, "shrinkB": 4},
    )


def diverging_norm(values: np.ndarray) -> TwoSlopeNorm:
    """Diverging scale pinned at zero, symmetric about it.

    Pinning matters: a colour map centred on the data's own mean would paint a
    uniformly positive panel half blue, which reads as "half of these are
    negative" to anyone who does not check the colour bar.
    """
    finite = values[np.isfinite(values)]
    extent = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    extent = max(extent, 1e-9)
    return TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent)


#: Blue-to-vermillion through near-white, built from the Okabe-Ito endpoints so
#: the heatmap shares the palette the rest of the repo uses.
DIVERGING = LinearSegmentedColormap.from_list(
    "okabe_diverging", [MISS, "#F2EFEA", BEAT], N=256
)


def save(fig: Figure, path: Path, *, raster: bool = False) -> Path:
    """Write a figure deterministically.

    Matplotlib stamps SVG output with the current date by default, which would
    make every regeneration a diff even when nothing changed. Suppressing it
    keeps the committed figures stable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if raster:
        fig.savefig(path, dpi=RASTER_DPI, format="png")
    else:
        fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)
    return path
