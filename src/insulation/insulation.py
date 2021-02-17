import itertools
import os

import bioframe
import click
import cooler
import cooltools
import cooltools.lib.plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cooltools.insulation import calculate_insulation_score, find_boundaries
from matplotlib.colors import LogNorm
from matplotlib.pyplot import flag
from matplotlib.ticker import EngFormatter
from mpl_toolkits.axes_grid import make_axes_locatable
from skimage.filters import threshold_li, threshold_otsu


@click.group()
def cli():
    pass


def plot_45_mat(ax, clr_mat, start=0, resolution=1000, *args, **kwargs):
    start_pos_vec = [start + resolution * i for i in range(len(clr_mat) + 1)]
    n = clr_mat.shape[0]
    t = np.array([[1, 0.5], [-1, 0.5]])
    matrix = np.dot(np.array([(i[1], i[0]) for i in itertools.product(start_pos_vec[::-1], start_pos_vec)]), t)
    x = matrix[:, 1].reshape(n + 1, n + 1)
    y = matrix[:, 0].reshape(n + 1, n + 1)
    img = ax.pcolormesh(x, y, np.flipud(clr_mat), *args, **kwargs)
    img.set_rasterized(True)

    return img


def format_ticks(ax, x=True, y=True, rotate=True):
    bp_formatter = EngFormatter('b')
    if y:
        ax.yaxis.set_major_formatter(bp_formatter)
    if x:
        ax.xaxis.set_major_formatter(bp_formatter)
        ax.xaxis.tick_bottom()
    if rotate:
        ax.tick_params(axis='x', rotation=45)


def plot_insulation(clr, insulation, windows, resolution, out_path, exclude_chroms, title):
    dir_path = os.path.join(os.path.dirname(out_path), title)

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    chromsizes = bioframe.fetch_chromsizes('sacCer3', filter_chroms=False)
    regions = [(k, 0, v) for k, v in chromsizes.drop('chrM').iteritems()]

    for region in regions:
        norm = LogNorm(vmax=0.1, vmin=0.001)
        data = clr.matrix(balance=True).fetch(region)
        fig, ax = plt.subplots(figsize=(20, 4))

        img = plot_45_mat(ax, data, start=0, resolution=resolution, norm=norm, cmap='fall')

        ax.set_aspect(0.5)
        ax.set_ylim(0, 30000)
        format_ticks(ax, rotate=False)
        ax.xaxis.set_visible(False)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='1%' ,pad=0.1, aspect=6)
        plt.colorbar(img, cax=cax)

        insul_region = bioframe.select(insulation, region)
        
        ins_ax = divider.append_axes('bottom', size='50%', pad=0.0, sharex=ax)
        ins_ax.set_prop_cycle(plt.cycler('color', plt.cm.plasma(np.linspace(0, 1, 5))))

        for window in windows:
            ins_ax.plot(insul_region[['start', 'end']].mean(axis=1),
                insul_region[f'log2_insulation_score_{window}'],
                label=f'{window} bp window', lw=1)

        ins_ax.legend(bbox_to_anchor=(1.125, 1.05), loc='upper right')
        fig.suptitle(f'{title}: {region[0]}')

        path = os.path.join(dir_path, '_'.join((region[0], os.path.basename(out_path))))

        plt.savefig(path, dpi=300)
    

@cli.command('compute-insulation')
@click.argument(
    'cooler_path', nargs=1,
    type=click.Path(exists=True), required=True
)
@click.option(
    '--out', '-o', 'out_path',
    type=click.Path(),
    help='The path to the output insulation score plot.'
)
@click.option(
    '--resolution', nargs=1,
    type=int, default=1000,
    help='Resolution of the Hi-C data in a .mcool file.'
)
@click.option(
    '--window', '-w', 'windows', multiple=True,
    type=int, default=1000,
    help=''
)
@click.option(
    '--exclude-chrom', 'exclude_chroms',
    type=str, multiple=True,
    help='Exclude the specified chromosome from the insulation score plot.'
)
@click.option(
    '--title', '-t', 'title',
    type=str, nargs=1,
    help='Title text for the output plot.'
)
def compute_insulation(
    cooler_path,
    out_path,
    windows,
    resolution,
    exclude_chroms,
    title
):
    clr = cooler.Cooler('::/resolutions/'.join((cooler_path, str(resolution))))
    insulation = calculate_insulation_score(clr, windows)
    plot_insulation(clr, insulation, windows, resolution, out_path, exclude_chroms, title)


def plot_boundary_strengths(insulation, windows, title, threshold, out_path):
    histkwargs = dict(
        bins=10**np.linspace(-4, 1, 100),
        histtype='step',
        lw=1
    )

    fig, axs = plt.subplots(len(windows), 1, sharex=True, figsize=(4, 6))

    for i, (w, ax) in enumerate(zip(windows, axs)):
        ax.hist(
            insulation[f'boundary_strength_{w}'],
            **histkwargs
        )
        ax.text(
            0.02, 0.9,
            f'{w//1000}kb window',
            ha='left',
            va='top',
            transform=ax.transAxes
        )
        ax.set(
            xscale='log',
            ylabel='# boundaries'
        )

        if threshold:
            th_li = threshold_li(insulation[f'boundary_strength_{w}'].dropna().values)
            th_otsu = threshold_otsu(insulation[f'boundary_strength_{w}'].dropna().values)
            n_boundaries_li = (insulation[f'boundary_strength_{w}'].dropna()>=th_li).sum()
            n_boundaries_otsu = (insulation[f'boundary_strength_{w}'].dropna()>=th_otsu).sum()

            ax.axvline(th_li, c='green')
            ax.axvline(th_otsu, c='magenta')

            ax.text(
                0.01, 0.7,
                f'{n_boundaries_otsu} boundaries (Otsu)',
                c='magenta',
                ha='left',
                va='top',
                transform=ax.transAxes
            )
            ax.text(
                0.01, 0.5,
                f'{n_boundaries_li} boundaries (Li)',
                c='green',
                ha='left',
                va='top',
                transform=ax.transAxes
            )

    axs[-1].set(xlabel='boundary strength')
    fig.suptitle(title)
    plt.savefig(out_path, dpi=300)


@cli.command('boundaries')
@click.argument(
    'cooler_path', nargs=1,
    type=click.Path(exists=True), required=True
)
@click.option(
    '--out', '-o', 'out_path',
    type=click.Path(),
    help='The path to the output insulation score plot.'
)
@click.option(
    '--resolution', nargs=1,
    type=int, default=1000,
    help='Resolution of the Hi-C data in a .mcool file.'
)
@click.option(
    '--window', '-w', 'windows', multiple=True,
    type=int, default=1000,
    help=''
)
@click.option(
    '--exclude-chrom', 'exclude_chroms',
    type=str, multiple=True,
    help='Exclude the specified chromosome from the insulation score plot.'
)
@click.option(
    '--threshold', is_flag=True, default=False,
    help='Calculate and plot thresholds for stronger boundaries using Otsu\'s and Li\'s algorithms from scikit-image. Default is false.'
)
@click.option(
    '--title', '-t', 'title',
    type=str, nargs=1,
    help='Title text for the output plot.'
)
def boundaries(
    cooler_path,
    out_path,
    windows,
    resolution,
    exclude_chroms,
    threshold,
    title
):
    clr = cooler.Cooler('::/resolutions/'.join((cooler_path, str(resolution))))
    insulation = calculate_insulation_score(clr, windows)
    boundaries = find_boundaries(insulation)
    plot_boundary_strengths(boundaries, windows, title, threshold, out_path)



if __name__ == "__main__":
    cli()
