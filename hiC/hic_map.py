from hiC.pileups import snip_pileup
import os

import click
from h5py._hl import base
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, LogLocator, LogFormatterSciNotation
from matplotlib import cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import numpy as np
import pandas as pd

import bioframe
import cooler
import cooltools.lib.plotting
import cooltools.api.snipping
from cooltools.api import snipping


def snip_pileup(clr, resolution, features, chromsizes, flank):
    windows = snipping.make_bin_aligned_windows(
        resolution, features["chrom"], features["mid"], flank_bp=flank
    )
    supports = chromsizes[["chrom", "start", "end"]].values
    windows = snipping.assign_regions(windows, supports)

    snipper = cooltools.snipping.CoolerSnipper(clr)
    stack = cooltools.snipping.pileup(windows, snipper.select, snipper.snip)
    stack = np.nanmean(stack, axis=2)

    return stack


def plot_pileup_map(
    contact_map,
    flank,
    window_size,
    resolution,
    ratio,
    title,
    font_size,
    out_path,
    logbase=10,
    vmin=0.0001,
    vmax=0.01,
    transparent=False,
):
    font = {"size": font_size}
    plt.rc("font", **font)

    fig, ax = plt.subplots(figsize=(7, 7), ncols=1)

    if ratio:
        colormap = "RdBu_r"
        cbar_label = "ratio means (log)"
    else:
        colormap = "fall"
        cbar_label = "mean contact frequency (log)"

    im = ax.matshow(contact_map, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=colormap)

    ax.set_xlabel("relative position (kb)")
    ax.set_ylabel("relative position (kb)")

    ticks_px = np.linspace(0, flank * 2 // resolution, 5)
    ticks_kbp = [int((pos - ticks_px[-1] / 2) * resolution // 1000) for pos in ticks_px]
    ticks_kbp[2] = "CEN"
    # ticks_kbp = [f'{pos} kb' for pos in ticks_kbp]
    ax.xaxis.set_ticks(ticks_px)
    ax.xaxis.set_ticklabels(ticks_kbp)
    ax.yaxis.set_ticks(ticks_px)
    ax.yaxis.set_ticklabels(ticks_kbp)
    ax.xaxis.tick_bottom()
    ax.tick_params(axis="x", rotation=45)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(
        im,
        cax=cax,
        label=cbar_label,
        ticks=LogLocator(base=logbase),
        format=LogFormatterSciNotation(base=logbase),
    )

    fig.suptitle(title)
    fig.subplots_adjust(top=0.5)

    plt.tight_layout()
    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        facecolor=None if transparent else "white",
        transparent=transparent,
    )
    plt.close()


def plot(
    contact_map,
    start,
    end,
    chrom_names,
    chrom_positions,
    bin_size,
    region,
    ratio,
    title,
    font_size,
    annotations,
    centromeres=None,
    out_path=None,
    logbase=10,
    plot_ticks=True,
    plot_colorbar=True,
    plot_chrom_labels=True,
    label_bar_pad=5,
    vmin=0.00001,
    vmax=0.1,
    transparent=False,
):
    font = {"size": font_size}
    plt.rc("font", **font)

    fig, ax = plt.subplots(figsize=(7, 7), ncols=1)

    if ratio:
        colormap = "RdBu_r"
        cbar_label = "ratio (log)"
    else:
        colormap = "fall"
        cbar_label = "contact frequency (log)"

    im = ax.matshow(
        contact_map,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap=colormap,
        extent=(start, end, end, start),
    )

    if centromeres is not None:
        centromeres = {cen.chrom: cen.mid // bin_size for cen in centromeres.itertuples()}

    if region[0] == region[1]:

        format_chrom_labels(
            ax=ax,
            chrom_names=chrom_names,
            chrom_positions=chrom_positions,
            centromeres=centromeres,
            label_bar_pad=label_bar_pad,
            plot_labels=plot_chrom_labels,
        )
        if plot_ticks:
            format_ticks(
                ax=ax, x=True, y=False, rotate=True, unit="Mb", bin_size=bin_size
            )
        else:
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        ax.set_xticks([pos[0] for pos in chrom_positions])
        ax.set_yticks([pos[0] for pos in chrom_positions])
        ax.grid(color="black")

        format_chrom_labels(
            ax=ax,
            chrom_names=chrom_names,
            chrom_positions=chrom_positions,
            centromeres=centromeres,
            label_bar_pad=label_bar_pad,
            plot_labels=plot_chrom_labels,
        )

    if annotations:
        format_annotations(ax, annotations, bin_size, 0)

    ax.set_xlim(start, end)
    ax.set_ylim(start, end)
    ax.invert_yaxis()

    if plot_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(
            im,
            cax=cax,
            label=cbar_label,
            ticks=LogLocator(base=logbase),
            format=LogFormatterSciNotation(base=logbase),
        )

    fig.suptitle(title)
    fig.subplots_adjust(top=0.13)

    plt.tight_layout()
    if out_path:
        plt.savefig(
            out_path,
            dpi=300,
            bbox_inches="tight",
            facecolor=None if transparent else "white",
            transparent=transparent,
        )
        plt.close()
    else:
        plt.plot()


def format_chrom_labels(
    ax,
    chrom_names,
    chrom_positions,
    centromeres=None,
    label_bar_pad=7,
    plot_labels=True
):
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    x_trans = ax.get_xaxis_transform()
    y_trans = ax.get_yaxis_transform()

    cmap = cm.summer(np.linspace(0, 1, len(chrom_names) * 2))

    for i, (chrom, (start, end)) in enumerate(zip(chrom_names, chrom_positions)):
        if plot_labels:
            # plot axis labels
            ax.annotate(
                chrom,
                xy=((start + end) // 2, 1.037),
                xycoords=x_trans,
                ha="center",
                va="bottom",
            )
            ax.annotate(
                chrom,
                xy=(-0.037, (start + end) // 2),
                xycoords=y_trans,
                ha="right",
                va="center",
                rotation=90,
            )

        # plot label bars
        ax.plot(
            [start + label_bar_pad, end - label_bar_pad],
            [1.02, 1.02],
            color=cmap[i],
            transform=x_trans,
            lw=7,
            clip_on=False,
        )
        ax.plot(
            [-0.02, -0.02],
            [start + label_bar_pad, end - label_bar_pad],
            color=cmap[i],
            transform=y_trans,
            lw=7,
            clip_on=False,
        )

        # plot centromere positions
        if centromeres:
            ax.plot(
                start + centromeres[chrom],
                1.02,
                transform=x_trans,
                color="black",
                marker="o",
                markersize=6,
                clip_on=False,
            )
            ax.plot(
                -0.02,
                start + centromeres[chrom],
                transform=y_trans,
                color="black",
                marker="o",
                markersize=6,
                clip_on=False,
            )


def format_annotations(ax, annotations, bin_size, label_bar_pad=0):
    x_trans = ax.get_xaxis_transform()
    y_trans = ax.get_yaxis_transform()

    cmap = cm.autumn(np.linspace(0, 1, len(annotations)))

    for i, (label, (start, end)) in enumerate(annotations):
        start = start // bin_size
        end = end // bin_size

        # plot labels
        ax.annotate(
            label,
            xy=((start + end) // 2, 1.037),
            xycoords=x_trans,
            ha="center",
            va="bottom",
        )
        ax.annotate(
            label,
            xy=(-0.037, (start + end) // 2),
            xycoords=y_trans,
            ha="right",
            va="center",
            rotation=90,
        )

        # plot label bars
        ax.plot(
            [start + label_bar_pad, end - label_bar_pad],
            [1.02, 1.02],
            color=cmap[i],
            transform=x_trans,
            lw=7,
            clip_on=False,
        )
        ax.plot(
            [-0.02, -0.02],
            [start + label_bar_pad, end - label_bar_pad],
            color=cmap[i],
            transform=y_trans,
            lw=7,
            clip_on=False,
        )


def format_ticks(ax, x=True, y=True, rotate=True, unit="b", bin_size=1):
    if unit == "b":
        div = 1
    elif unit == "kb":
        div = 1000
    elif unit == "Mb":
        div = 100_000

    bp_formatter = FuncFormatter(lambda x, pos: f"{int(x * bin_size // div)}")
    if y:
        ax.yaxis.set_major_formatter(bp_formatter)
        ax.set_ylabel(f"position ({unit})")
    else:
        ax.yaxis.set_ticks([])
    if x:
        ax.xaxis.set_major_formatter(bp_formatter)
        ax.set_xlabel(f"position ({unit})")
        ax.xaxis.tick_bottom()
    else:
        ax.xaxis.set_ticks([])
    if rotate:
        ax.tick_params(axis="x", rotation=45)


def get_contact_map(clr, balanced, start=None, end=None):
    if start is None or end is None:
        return clr.matrix(balance=balanced)[:]
    else:
        return clr.matrix(balance=balanced)[start:end, start:end]


def get_centromeres(centromeres_path, assembly):
    if centromeres_path:
        centromeres = pd.read_csv(
            centromeres_path,
            delim_whitespace=True,
            header=None,
            names=["chrom", "start", "end", "mid"],
        )
        centromeres["mid"] = centromeres.apply(
            lambda row: (row["start"] + row["end"]) // 2, axis=1
        )
    else:
        centromeres = bioframe.fetch_centromeres(assembly)
        chromsizes = bioframe.fetch_chromsizes(
            assembly, filter_chroms=False, as_bed=True
        )
        centromeres = bioframe.core.construction.add_ucsc_name_column(
            bioframe.make_chromarms(chromsizes, centromeres)
        )

    return centromeres


def get_chromsizes(assembly):
    return bioframe.fetch_chromsizes(assembly, filter_chroms=False, as_bed=True)


def open_cooler(path, resolution):
    ext = os.path.splitext(path)[1]
    if ext == ".cool":
        return cooler.Cooler(path)
    elif ext == ".mcool":
        return cooler.Cooler("::/resolutions/".join((path, str(resolution))))
    else:
        raise TypeError("Provided file is not of required .cool or .mcool type.")


@click.group()
def cli():
    pass


@cli.command("plot-map")
@click.argument("cooler-path", nargs=1, type=click.Path(exists=True), required=True)
@click.option(
    "--out",
    "-o",
    "out_path",
    type=click.Path(),
    required=True,
    help="Path to the output file.",
)
@click.option(
    "--resolution",
    nargs=1,
    type=int,
    default=1000,
    help="Resolution of the Hi-C Data (if a multi-resolution cooler was provided).",
)
@click.option(
    "--region",
    "-r",
    type=str,
    nargs=2,
    help="Region to plot, given in chromosome names for the start and end regions. If omitted the entire genome will be plotted.",
)
@click.option(
    "--exclude-chrom",
    "exclude_chroms",
    type=str,
    multiple=True,
    help="Exclude the specified chromosome from the pileups.",
)
@click.option(
    "--balanced",
    "-b",
    is_flag=True,
    help="Plot the balanced contact map. Only works with .multicoolers.",
)
@click.option(
    "--assembly",
    "-a",
    type=str,
    nargs=1,
    help="Assembly name to be used for downloading chromsizes.",
)
@click.option(
    "--chromsizes",
    "chromsizes_path",
    type=click.Path(exists=True),
    help="Path to a text file containing chromsizes. If not provided, chromsizes will be downloaded based on --assembly.",
)
@click.option(
    "--centromeres",
    "centromeres_path",
    type=click.Path(exists=True),
    help="Path to a text file containing centromere start and end positions. If provided these coordinates will be used instead of --region to create a pileup of centromeres.",
)
@click.option("--title", type=str, nargs=1, help="Title text for the pileup plot.")
@click.option(
    "--font-size",
    type=int,
    default=16,
    help="Font size of the plot's labels and title text. Default is 16.",
)
def plot_map(
    cooler_path,
    out_path,
    resolution,
    region,
    exclude_chroms,
    balanced,
    assembly,
    chromsizes_path,
    centromeres_path,
    title,
    font_size,
):
    centromeres = get_centromeres(centromeres_path, assembly)

    clr = open_cooler(cooler_path, resolution)
    start = clr.extent(region[0])[0]
    end = clr.extent(region[1])[1]

    contact_map = get_contact_map(clr, balanced, start, end)

    chrom_names = clr.chromnames[
        clr.chromnames.index(region[0]) : clr.chromnames.index(region[1]) + 1
    ]
    chrom_positions = [clr.extent(chrom) for chrom in chrom_names]

    plot(
        contact_map,
        start,
        end,
        chrom_names,
        chrom_positions,
        clr.binsize,
        centromeres,
        region,
        False,
        title,
        font_size,
        out_path,
    )


@cli.command("plot-ratio")
@click.argument("cooler-paths", nargs=2, type=click.Path(exists=True), required=True)
@click.option(
    "--out",
    "-o",
    "out_path",
    type=click.Path(),
    required=True,
    help="Path to the output file.",
)
@click.option(
    "--resolution",
    nargs=1,
    type=int,
    default=1000,
    help="Resolution of the Hi-C Data (if a multi-resolution cooler was provided).",
)
@click.option(
    "--region",
    "-r",
    type=str,
    nargs=2,
    help="Region to plot, given in chromosome names for the start and end regions. If omitted the entire dataset will be plotted.",
)
@click.option(
    "--plot-single-maps",
    is_flag=True,
    default=False,
    help="Plot single contact maps of the provided .coolers to the side of the ratio map.",
)
@click.option(
    "--exclude-chrom",
    "exclude_chroms",
    type=str,
    multiple=True,
    help="Exclude the specified chromosome from the pileups.",
)
@click.option("--balanced", "-b", is_flag=True, help="Plot the balanced contact map.")
@click.option(
    "--assembly",
    "-a",
    type=str,
    nargs=1,
    required=True,
    help="Assembly name to be used for downloading chromsizes.",
)
@click.option(
    "--chromsizes",
    "chromsizes_path",
    type=click.Path(exists=True),
    help="Path to a text file containing chromsizes. If not provided, chromsizes will be downloaded based on --assembly.",
)
@click.option(
    "--centromeres",
    "centromeres_path",
    type=click.Path(exists=True),
    help="Path to a text file containing centromere start and end positions. If provided these coordinates will be used instead of --region to create a pileup of centromeres.",
)
@click.option("--title", type=str, help="Title text for the pileup plot.")
@click.option(
    "--font-size",
    type=int,
    default=16,
    help="Font size of the plot's labels and title text. Default is 16.",
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
    title,
    font_size,
):
    centromeres = get_centromeres(centromeres_path, assembly)

    clrs = [open_cooler(path, resolution) for path in cooler_paths]
    start = clrs[0].extent(region[0])[0]
    end = clrs[0].extent(region[1])[1]

    contact_maps = [get_contact_map(clr, balanced, start, end) for clr in clrs]
    contact_map = contact_maps[0] / contact_maps[1]

    chrom_names = clrs[0].chromnames[
        clrs[0].chromnames.index(region[0]) : clrs[0].chromnames.index(region[1]) + 1
    ]
    chrom_positions = [clrs[0].extent(chrom) for chrom in chrom_names]

    plot(
        contact_map,
        start,
        end,
        chrom_names,
        chrom_positions,
        clrs[0].binsize,
        centromeres,
        region,
        True,
        title,
        font_size,
        out_path,
    )


@cli.command("plot-pileup")
@click.argument("cooler-path", nargs=1, type=click.Path(exists=True), required=True)
@click.option(
    "--out",
    "-o",
    "out_path",
    type=click.Path(),
    required=True,
    help="Path to the output file.",
)
@click.option(
    "--resolution",
    nargs=1,
    type=int,
    default=1000,
    help="Resolution of the Hi-C Data (if a multi-resolution cooler was provided).",
)
@click.option(
    "--region",
    "-r",
    type=str,
    default="centromere",
    help='Center genomic coordinates of a quadratic window to create a pileup of. Use "centromere" to plot pileups of centromeres (default).',
)
@click.option(
    "--window-size",
    type=int,
    default=200_000,
    help="Size of the quadratic pileup window sides in bp. Default is 200,000 bp.",
)
@click.option(
    "--exclude-chrom",
    "exclude_chroms",
    type=str,
    multiple=True,
    help="Exclude the specified chromosome from the pileups.",
)
@click.option(
    "--balanced",
    "-b",
    is_flag=True,
    help="Plot the balanced contact map. Only works with .multicoolers.",
)
@click.option(
    "--assembly",
    "-a",
    type=str,
    nargs=1,
    help="Assembly name to be used for downloading chromsizes.",
)
@click.option(
    "--chromsizes",
    "chromsizes_path",
    type=click.Path(exists=True),
    help="Path to a text file containing chromsizes. If not provided, chromsizes will be downloaded based on --assembly.",
)
@click.option(
    "--centromeres",
    "centromeres_path",
    type=click.Path(exists=True),
    help="Path to a text file containing centromere start and end positions. If provided these coordinates will be used instead of --region to create a pileup of centromeres.",
)
@click.option("--title", type=str, nargs=1, help="Title text for the pileup plot.")
@click.option(
    "--font-size",
    type=int,
    default=16,
    help="Font size of the plot's labels and title text. Default is 16.",
)
def plot_pileup(
    cooler_path,
    out_path,
    resolution,
    region,
    window_size,
    exclude_chroms,
    balanced,
    assembly,
    chromsizes_path,
    centromeres_path,
    title,
    font_size,
):
    """
    Plots pileups of a specified window size around a genomic region for an input cooler file.
    """
    centromeres = get_centromeres(centromeres_path, assembly)
    chromsizes = get_chromsizes(assembly)
    chromsizes = chromsizes[~chromsizes.chrom.isin(exclude_chroms)]

    clr = open_cooler(cooler_path, resolution)

    flank = window_size // 2
    contact_map = snip_pileup(clr, resolution, centromeres, chromsizes, flank)

    plot_pileup(
        contact_map, flank, window_size, resolution, False, title, font_size, out_path
    )


if __name__ == "__main__":
    np.seterr(divide="ignore", invalid="ignore")
    cli()
