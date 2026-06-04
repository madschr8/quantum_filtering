"""Thesis-matched matplotlib styling. Made with a good amount of help from Claude Opus 4.8.

Importing this module sets rcParams to render plots in Computer Modern (the
scrreprt default body font) via LaTeX (text.usetex=True). A LaTeX install is
required -- import raises if 'latex' is not on PATH rather than silently
falling back to mathtext, so the figures always carry the thesis aesthetic.
Body text is 10 pt and figures are sized to the thesis \\linewidth (16 cm =
6.3 in).

    import plot_style as ps
    fig, ax = plt.subplots(figsize=ps.figsize())
    ax.plot(..., color=ps.SEMANTIC['model'])
"""

import os
import shutil
import matplotlib as mpl
from cycler import cycler

# This module hard-requires LaTeX: text.usetex=True is the only way the figures
# match the thesis's Computer Modern aesthetic exactly, so a missing install is
# a hard error rather than a silent mathtext fallback that would look different.
if shutil.which('latex') is None:
    raise RuntimeError(
        "plot_style requires a LaTeX install on PATH for text.usetex=True, but "
        "'latex' was not found. Install TeX Live or MiKTeX (including dvipng) "
        "and reopen the shell, or import a non-usetex style instead. Refusing "
        "to fall back to mathtext so the figures keep the right aesthetic."
    )

# Punctuation spelled the LaTeX way (usetex is always active in this module):
#   f'95{PCT} CI'            ->  '95\\% CI'
#   f'Plot 2 {EMDASH} title' ->  em dash (LaTeX renders '---' as an em dash)
#   f'trajectory {HASH}{i}'  ->  escaped '#'
EMDASH = '---'
PCT    = r'\%'
HASH   = r'\#'
AMP    = r'\&'

# Thesis text width is 16 cm = 6.3 in (\linewidth). Every figure is sized to
# this so it fills the column exactly; FULLWIDTH matches it (no margin overflow).
TEXTWIDTH_IN = 6.3
FULLWIDTH_IN = 6.3
GOLDEN       = (1 + 5 ** 0.5) / 2


def figsize(width=TEXTWIDTH_IN, aspect=GOLDEN, height=None):
    """Figure size in inches. Pass `aspect` (=width/height) or explicit `height`."""
    if height is not None:
        return (width, height)
    return (width, width / aspect)


# Directory (relative to the notebook) where saved figures land.
FIG_DIR = 'figures'


def save_fig(fig, name, fmt='pdf', dpi=300):
    """Save `fig` to FIG_DIR/<name>.<fmt> for inclusion in the thesis.

    Writes a vector PDF (fmt='pdf' by default) with bbox_inches='tight' so
    LaTeX gets tight margins; `dpi` only sets the resolution of any rasterised
    inset (there are none here, so it is just a sensible fallback). Returns the
    written path.
    """
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f'{name}.{fmt}')
    fig.savefig(path, format=fmt, dpi=dpi, bbox_inches='tight')
    return path


PALETTE = {
    'red':    '#B01720',   # UCPH main
    'red_dk': '#8F2424',   # UCPH accessory
    'blue':   '#1F4E79',
    'green':  '#2E8B57',
    'teal':   '#2C7873',
    'ochre':  '#C68A12',
    'purple': '#5D478B',
    'gray':   '#4D4D4D',
    'gray_l': '#9A9A9A',
}

# Blue/red/green are reserved for rho plotting; diagnostics use purple/ochre/teal.
SEMANTIC = {
    'truth':     PALETTE['blue'],
    'model':     PALETTE['red'],
    'fit':       PALETTE['green'],
    'reference': PALETTE['gray'],
    'aux':       PALETTE['teal'],
}

_CYCLE = [PALETTE['red'], PALETTE['blue'], PALETTE['teal'],
          PALETTE['ochre'], PALETTE['purple'], PALETTE['red_dk']]

# No font package -> LaTeX keeps its Computer Modern default (matches scrreprt).
_LATEX_PREAMBLE = r"""
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}
"""

mpl.rcParams.update({
    'text.usetex':         True,
    'text.latex.preamble': _LATEX_PREAMBLE,
    'font.family':         'serif',
    'font.serif':          ['Computer Modern Roman', 'CMU Serif',
                            'Latin Modern Roman'],
    # Computer Modern math so any mathtext-rendered $\rho$, $\Omega$, ... still
    # matches the body font (usetex handles the rest).
    'mathtext.fontset':    'cm',

    # Uniform 9 pt on every figure element -- equals \small (the caption size in
    # a 10 pt KOMA class), so figure text sits cohesively just below the 10 pt
    # body and matches the captions. Set once here; cells must not override it
    # per-figure (otherwise text size drifts between figures).
    'font.size':         9,
    'axes.titlesize':    9,
    'axes.labelsize':    9,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
    'figure.titlesize':  9,

    'axes.prop_cycle':   cycler(color=_CYCLE),

    'lines.linewidth':   1.1,
    'lines.markersize':  4.0,
    'patch.linewidth':   0.6,

    'axes.linewidth':    0.8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.titlepad':     6.0,
    'axes.labelpad':     4.0,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size':  3.0,
    'ytick.major.size':  3.0,
    'xtick.direction':   'out',
    'ytick.direction':   'out',

    'axes.grid':       True,
    'grid.linewidth':  0.4,
    'grid.alpha':      0.5,
    'grid.color':      PALETTE['gray_l'],

    'legend.frameon':       False,
    'legend.borderpad':     0.3,
    'legend.handlelength':  1.6,
    'legend.handletextpad': 0.6,
    'legend.labelspacing':  0.3,

    'figure.figsize':     (TEXTWIDTH_IN, TEXTWIDTH_IN / GOLDEN),
    'figure.dpi':         120,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.02,
    'savefig.format':     'svg',
    'pdf.fonttype':       42,
})
