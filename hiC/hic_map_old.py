from genericpath import exists
import os
from typing import Type

import click
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import EngFormatter
from matplotlib import cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import numpy as np
import pandas as pd

import bioframe
import cooler
import cooltools.lib.plotting


def plot(contact_map, start, end, clr, centromeres, chroms, title, out_path):
    fig, ax = plt.subplots(
        figsize=(7, 7),
        ncols=1
    )
    im = ax.matshow(
        contact_map,
        norm=LogNorm(vmin=0.00001, vmax=0.1),
        cmap='fall',
        extent=(start, end, end, start)
    )

    chrom_names = clr.chromnames[clr.chromnames.index(chroms[0]) : clr.chromnames.index(chroms[1]) + 1]
    chrom_positions = [clr.extent(chrom) for chrom in chrom_names]

    ax.set_xticks([pos[0] for pos in chrom_positions])
    ax.set_yticks([pos[0] for pos in chrom_positions])

    ax.set_xlim(start, end)
    ax.set_ylim(start, end)
    ax.invert_yaxis()
    ax.grid(color='black')

    if chroms[0] == chroms[1]:
        format_ticks(ax, True, True, True)
    else:
        format_chrom_labels(ax, chrom_names=chrom_names, chrom_positions=chrom_positions, centromeres={chrom: pos // clr.binsize for chrom, pos in centromeres.items()})

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label='contact frequency (log10)')

    fig.suptitle(title)
    fig.subplots_adjust(top=0.78)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
    plt.close()


def format_chrom_labels(ax, chrom_names, chrom_positions, centromeres):
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    x_trans = ax.get_xaxis_transform()
    y_trans = ax.get_yaxis_transform()

    cmap = cm.summer(np.linspace(0, 1, len(chrom_names) * 2))

    pad = 7

    for i, (chrom, (start, end)) in enumerate(zip(chrom_names, chrom_positions)):
        # plot axis labels
        ax.annotate(chrom, xy=((start + end) // 2, 1.06), xycoords=x_trans, ha='center', va='top')
        ax.annotate(chrom, xy=(-0.048, (start + end) // 2 - 20), xycoords=y_trans, ha='center', va='top', rotation=90)

        # plot label bars
        ax.plot([start + pad, end - pad], [1.02, 1.02], color=cmap[i], transform=x_trans, lw=7, clip_on=False)
        ax.plot([-0.02, -0.02], [start + pad, end - pad], color=cmap[i], transform=y_trans, lw=7, clip_on=False)

        # plot centromere positions
        ax.plot(start + centromeres[chrom], 1.02, transform=x_trans, color='black', marker='o', markersize=6, clip_on=False)
        ax.plot(-0.02, start + centromeres[chrom], transform=y_trans, color='black', marker='o', markersize=6, clip_on=False)


def format_ticks(ax, x=True, y=True, rotate=True):
    bp_formatter = EngFormatter('b')
    if y:
        ax.yaxis.set_major_formatter(bp_formatter)
    if x:
        ax.xaxis.set_major_formatter(bp_formatter)
        ax.xaxis.tick_bottom()
    if rotate:
        ax.tick_params(axis='x',rotation=45)


def get_contact_map(clr, balanced, start=None, end=None):
    if start is None or end is None:
        return clr.matrix(balance=balanced)[:]
    else:
        return clr.matrix(balance=balanced)[start:end, start:end]


def get_centromeres(centromeres_path, assembly):
    if centromeres_path:
        centromeres = {r[1][0]: (r[1][1] + r[1][2]) // 2 for r in pd.read_csv(centromeres_path, delim_whitespace=True, header=None).iterrows()}
    else:
        centromeres = bioframe.fetch_centromeres(assembly)

    return centromeres


def open_cooler(path, resolution):
    ext = os.path.splitext(path)[1]
    if ext == '.cool':
        return cooler.Cooler(path)
    elif ext == '.mcool':
        return cooler.Cooler('::/resolutions/'.join((path, str(resolution))))
    else:
        raise TypeError('Provided file is not of required .cool or .mcool type.')


@click.group()
def cli():
    pass


@cli.command('plot-map')
@click.argument(
    'cooler-path', nargs=1,
    type=click.Path(exists=True), required=True
)
@click.option(
    '--out', '-o', 'out_path',
    type=click.Path(), required=True,
    help='Path to the output file.'
)
@click.option(
    '--resolution', nargs=1,
    type=int, default=1000,
    help='Resolution of the Hi-C Data (if a multi-resolution cooler was provided).'
)
@click.option(
    '--chroms', type=str, nargs=2,
    help='Chromosomes to plot. If omitted the entire dataset will be plotted.'
)
@click.option(
    '--exclude-chrom', 'exclude_chroms',
    type=str, multiple=True,
    help='Exclude the specified chromosome from the pileups.'
)
@click.option(
    '--balanced', '-b', is_flag=True,
    help='Plot the balanced contact map. Only works with .multicoolers.'
)
@click.option(
    '--assembly', '-a',
    type=str, nargs=1,
    help='Assembly name to be used for downloading chromsizes.'
)
@click.option(
    '--chromsizes', 'chromsizes_path',
    type=click.Path(exists=True),
    help='Path to a text file containing chromsizes. If not provided, chromsizes will be downloaded based on --assembly.'
)
@click.option(
    '--centromeres', 'centromeres_path',
    type=click.Path(exists=True),
    help='Path to a text file containing centromere start and end positions.' 
)
@click.option(
    '--title', type=str, nargs=1,
    help='Title text for the pileup plot.'
)
def plot_map(
    cooler_path,
    out_path,
    resolution,
    chroms,
    exclude_chroms,
    balanced,
    assembly,
    chromsizes_path,
    centromeres_path,
    title
):
    clr = open_cooler(cooler_path, resolution)

    centromeres = get_centromeres(centromeres_path, assembly)

    start = clr.extent(chroms[0])[0]
    end = clr.extent(chroms[1])[1]

    contact_map = get_contact_map(clr, balanced, start, end)

    plot(contact_map, start, end, clr, centromeres, chroms, title, out_path)


@cli.command('plot-ratio')
@click.argument(
    'cooler-paths', nargs=2,
    type=click.Path(exists=True), required=True
)
@click.option(
    '--out', '-o', 'out_path',
    type=click.Path(), required=True,
    help='Path to the output file.'
)
@click.option(
    '--resolution', nargs=1,
    type=int, default=1000,
    help='Resolution of the Hi-C Data (if a multi-resolution cooler was provided).'
)
@click.option(
    '--region', '-r', type=str,
    help='Center coordinates of the genomic region to create a pileup of. Not used if --centromeres is provided.'
)
@click.option(
    '--plot-single-maps', is_flag=True, default=False,
    help='Plot single contact maps of the provided .coolers to the side of the ratio map.'
)
@click.option(
    '--exclude-chrom', 'exclude_chroms',
    type=str, multiple=True,
    help='Exclude the specified chromosome from the pileups.'
)
@click.option(
    '--balanced', '-b', is_flag=True,
    help='Plot the balanced contact map.'
)
@click.option(
    '--assembly', '-a',
    type=str, nargs=1, required=True,
    help='Assembly name to be used for downloading chromsizes.'
)
@click.option(
    '--chromsizes', 'chromsizes_path',
    type=click.Path(exists=True),
    help='Path to a text file containing chromsizes. If not provided, chromsizes will be downloaded based on --assembly.'
)
@click.option(
    '--centromeres', 'centromeres_path',
    type=click.Path(exists=True),
    help='Path to a text file containing centromere start and end positions. If provided these coordinates will be used instead of --region to create a pileup of centromeres.' 
)
@click.option(
    '--title', type=str,
    help='Title text for the pileup plot.'
)
def plot_ratio(
    cooler_paths,
    out_path,
    resolution,
    region,
    plot_single_maps,
    exclude_chroms,
    balanced,
    assembly,
    chromsizes_path,
    centromeres_path,
    title
):
    clrs = [open_cooler(path, resolution) for path in cooler_paths]

    chromsizes = get_centromeres(centromeres_path, assembly)

    chromsizes = chromsizes[~chromsizes.chrom.isin(exclude_chroms)]
    


if __name__ == '__main__':
    np.seterr(divide='ignore', invalid='ignore')
    cli()