"""Minimal SVG plotting, standard library only.

Why this exists rather than matplotlib: matplotlib's compiled extension
(``_c_internal_utils``) is blocked by an Application Control policy on the
development host, so importing it fails outright. Rather than make the map
visualiser and the benchmark charts depend on something that cannot load, both
render SVG directly.

That turns out to be the better choice regardless. SVG needs no native code, so
it works on any machine including the ESP32-era Raspberry Pi a reviewer might
use; it scales without resampling in a report; and the dashboard can inline it
next to the Canvas fleet view without a second rendering stack.

Coordinates are supplied in data space and mapped to SVG user units on write, so
callers never do their own arithmetic. SVG's y axis grows downward, which is
flipped here so data-space y grows upward as a reader expects.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from pathlib import Path


def _fmt(value: float) -> str:
    """Format a number compactly: no trailing zeros, no scientific notation."""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


@dataclass
class Canvas:
    """An SVG drawing surface with a data-space to pixel-space mapping."""

    width: int
    height: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    pad: int = 60
    background: str = "#FFFFFF"
    _parts: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        # Guard against a degenerate range, which would otherwise divide by zero
        # for a single-column map or a single-bar chart.
        if self.xmax <= self.xmin:
            self.xmax = self.xmin + 1
        if self.ymax <= self.ymin:
            self.ymax = self.ymin + 1

    # -- coordinate mapping --------------------------------------------------

    @property
    def plot_width(self) -> float:
        return self.width - 2 * self.pad

    @property
    def plot_height(self) -> float:
        return self.height - 2 * self.pad

    def sx(self, x: float) -> float:
        span = self.xmax - self.xmin
        return self.pad + (x - self.xmin) / span * self.plot_width

    def sy(self, y: float) -> float:
        span = self.ymax - self.ymin
        # Flipped: data y grows up, SVG y grows down.
        return self.height - self.pad - (y - self.ymin) / span * self.plot_height

    # -- primitives ----------------------------------------------------------

    def line(
        self,
        x1: float, y1: float, x2: float, y2: float,
        *,
        colour: str = "#333333",
        width: float = 1.0,
        dash: str | None = None,
        opacity: float = 1.0,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self._parts.append(
            f'<line x1="{_fmt(self.sx(x1))}" y1="{_fmt(self.sy(y1))}" '
            f'x2="{_fmt(self.sx(x2))}" y2="{_fmt(self.sy(y2))}" '
            f'stroke="{colour}" stroke-width="{_fmt(width)}" '
            f'stroke-linecap="round" opacity="{_fmt(opacity)}"{dash_attr}/>'
        )

    def polyline(
        self,
        points: list[tuple[float, float]],
        *,
        colour: str = "#1F77B4",
        width: float = 2.0,
        opacity: float = 1.0,
    ) -> None:
        if len(points) < 2:
            return
        coords = " ".join(
            f"{_fmt(self.sx(x))},{_fmt(self.sy(y))}" for x, y in points
        )
        self._parts.append(
            f'<polyline points="{coords}" fill="none" stroke="{colour}" '
            f'stroke-width="{_fmt(width)}" stroke-linejoin="round" '
            f'stroke-linecap="round" opacity="{_fmt(opacity)}"/>'
        )

    def circle(
        self,
        x: float, y: float, radius: float,
        *,
        fill: str = "#4C78A8",
        stroke: str = "#222222",
        stroke_width: float = 0.8,
    ) -> None:
        self._parts.append(
            f'<circle cx="{_fmt(self.sx(x))}" cy="{_fmt(self.sy(y))}" '
            f'r="{_fmt(radius)}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{_fmt(stroke_width)}"/>'
        )

    def square(
        self,
        x: float, y: float, size: float,
        *,
        fill: str = "#4C78A8",
        stroke: str = "#222222",
        stroke_width: float = 0.8,
    ) -> None:
        px, py = self.sx(x), self.sy(y)
        self._parts.append(
            f'<rect x="{_fmt(px - size)}" y="{_fmt(py - size)}" '
            f'width="{_fmt(2 * size)}" height="{_fmt(2 * size)}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{_fmt(stroke_width)}"/>'
        )

    def triangle(
        self,
        x: float, y: float, size: float,
        *,
        fill: str = "#4C78A8",
        stroke: str = "#222222",
        stroke_width: float = 0.8,
    ) -> None:
        px, py = self.sx(x), self.sy(y)
        pts = (
            f"{_fmt(px)},{_fmt(py - size)} "
            f"{_fmt(px - size)},{_fmt(py + size)} "
            f"{_fmt(px + size)},{_fmt(py + size)}"
        )
        self._parts.append(
            f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{_fmt(stroke_width)}"/>'
        )

    def rect_px(
        self,
        x: float, y: float, width: float, height: float,
        *,
        fill: str = "#4C78A8",
        stroke: str = "none",
        stroke_width: float = 1.0,
        opacity: float = 1.0,
    ) -> None:
        """Rectangle in pixel space. Used for chart bars and legend boxes."""
        self._parts.append(
            f'<rect x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(width)}" '
            f'height="{_fmt(height)}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{_fmt(stroke_width)}" opacity="{_fmt(opacity)}"/>'
        )

    def text(
        self,
        x: float, y: float, content: str,
        *,
        size: float = 11,
        colour: str = "#222222",
        anchor: str = "middle",
        weight: str = "normal",
        data_space: bool = True,
        dy: float = 0.0,
    ) -> None:
        px = self.sx(x) if data_space else x
        py = (self.sy(y) if data_space else y) + dy
        self._parts.append(
            f'<text x="{_fmt(px)}" y="{_fmt(py)}" font-size="{_fmt(size)}" '
            f'fill="{colour}" text-anchor="{anchor}" font-weight="{weight}" '
            f'font-family="system-ui, -apple-system, Segoe UI, sans-serif">'
            f"{html.escape(content)}</text>"
        )

    def multiline_text_px(
        self,
        x: float, y: float, lines: list[str],
        *,
        size: float = 11,
        colour: str = "#222222",
        line_height: float = 1.35,
        anchor: str = "start",
    ) -> None:
        for index, line in enumerate(lines):
            self.text(
                x, y + index * size * line_height, line,
                size=size, colour=colour, anchor=anchor, data_space=False,
            )

    def legend_px(self, x: float, y: float, lines: list[str], *, size: float = 11) -> None:
        """A boxed legend at a pixel-space position."""
        if not lines:
            return
        width = max(len(line) for line in lines) * size * 0.55 + 18
        height = len(lines) * size * 1.35 + 12
        self.rect_px(
            x, y, width, height,
            fill="#FFFFFF", stroke="#CCCCCC", stroke_width=1, opacity=0.9,
        )
        self.multiline_text_px(x + 9, y + size + 5, lines, size=size)

    def grid(self, *, x_ticks: list[float] | None = None,
             y_ticks: list[float] | None = None, colour: str = "#E8E8E8") -> None:
        for x in x_ticks or ():
            self.line(x, self.ymin, x, self.ymax, colour=colour, width=0.8)
        for y in y_ticks or ():
            self.line(self.xmin, y, self.xmax, y, colour=colour, width=0.8)

    # -- output --------------------------------------------------------------

    def to_svg(self, title: str = "") -> str:
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" '
            f'height="{self.height}" viewBox="0 0 {self.width} {self.height}">'
        )
        title_tag = f"<title>{html.escape(title)}</title>" if title else ""
        bg = (
            f'<rect width="{self.width}" height="{self.height}" '
            f'fill="{self.background}"/>'
        )
        body = "\n".join(self._parts)
        return f"{head}{title_tag}\n{bg}\n{body}\n</svg>\n"

    def save(self, path: str | Path, title: str = "") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_svg(title), encoding="utf-8")
        return path


def nice_ticks(low: float, high: float, target: int = 6) -> list[float]:
    """Round tick positions spanning ``low``..``high``.

    Picks a step from the 1/2/5 series so labels read as round numbers rather
    than as whatever the data range divided by six happened to produce.
    """
    if high <= low:
        return [low]
    raw_step = (high - low) / max(1, target)
    magnitude = 10 ** int(f"{raw_step:e}".split("e")[1])
    for multiple in (1, 2, 5, 10):
        step = multiple * magnitude
        if raw_step <= step:
            break
    start = (int(low / step)) * step
    ticks = []
    value = start
    while value <= high + step * 0.5:
        if value >= low - step * 0.5:
            ticks.append(round(value, 10))
        value += step
    return ticks
