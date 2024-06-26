import os
import sys

import bioframe
import click
import cooler
import cooltools
import cooltools.expected
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pairlib
import pairlib.scalings
import pairtools
from diskcache import Cache

# peri-centromeric/-telomeric region to remove from both sides of chromosomal arms
cache = Cache("~/.hic.cache")


@click.group()
def cli():
    pass


def plot_scalings(
    scalings, avg_trans_levels, plot_slope, label_subplots, labels, title, out_path
):
    """
    Plot scaling curves from a list of (bin, pair frequencies) tuples.
    """
    fig = plt.figure(constrained_layout=False)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 2], wspace=0.4, figure=fig)
    scale_ax = fig.add_subplot(gs[0, 0])
    slope_ax = fig.add_subplot(gs[0, 1]) if plot_slope else None

    for idx, scalings in enumerate(scalings):
        dist_bin_mids, pair_frequencies = scalings

        scale_ax.loglog(dist_bin_mids, pair_frequencies, label=labels[idx], lw=1)

        if avg_trans_levels:
            scale_ax.axhline(
                avg_trans_levels[idx],
                ls="dotted",
                c=scale_ax.get_lines()[-1].get_color(),
                lw=1,
            )

        if slope_ax is not None:
            slope_ax.semilogx(
                np.sqrt(dist_bin_mids.values[1:] * dist_bin_mids.values[:-1]),
                np.diff(np.log10(pair_frequencies.values))
                / np.diff(np.log10(dist_bin_mids.values)),
                label=labels[idx],
                lw=1,
            )

    scale_ax.grid(lw=0.5, color="gray")
    scale_ax.set_aspect(1.0)
    scale_ax.set_xlim(1e3, 1e6)
    scale_ax.set_ylim(0.0001, 2.0)
    scale_ax.set_xlabel("genomic separation (bp)")
    scale_ax.set_ylabel("contact frequency")
    scale_ax.set_anchor("S")

    handles, labels = scale_ax.get_legend_handles_labels()
    if avg_trans_levels:
        handles.append(Line2D([0], [0], color="black", lw=1, ls="dotted"))
        labels.append("average trans")

    scale_ax.legend(
        handles, labels, loc="upper left", bbox_to_anchor=(1.1, 1.0), frameon=False
    )

    if slope_ax is not None:
        slope_ax.grid(lw=0.5, color="gray")
        slope_ax.set_xlim(1e3, 1e6)
        slope_ax.set_ylim(-3.0, 0.0)
        slope_ax.set_yticks(np.arange(-3, 0.5, 0.5))
        slope_ax.set_aspect(1.0)
        slope_ax.set_xlabel("distance (bp)")
        slope_ax.set_ylabel("log-log slope")
        slope_ax.set_anchor("S")

    if label_subplots:
        scale_ax.set_title("(a)")
        slope_ax.set_title("(b)")

    fig.suptitle(title)

    plt.savefig(
        out_path, dpi=300, bbox_inches="tight", facecolor="white", transparent=False
    )
    plt.show()

    plt.close()


def open_pairs_file(path: str) -> pd.DataFrame:
    header, pairs_body = pairtools._headerops.get_header(
        pairtools._fileio.auto_open(path, "r")
    )
    cols = pairtools._headerops.extract_column_names(header)

    return pd.read_csv(pairs_body, header=None, names=cols, sep="\t")


def calc_pair_freqs(scalings, trans_levels, calc_avg_trans, normalized):
    dist_bin_mids = np.sqrt(scalings.min_dist * scalings.max_dist)
    pair_frequencies = scalings.n_pairs / scalings.n_bp2
    mask = pair_frequencies > 0

    avg_trans = None
    if calc_avg_trans:
        avg_trans = (
            trans_levels.n_pairs.astype("float64").sum()
            / trans_levels.np_bp2.astype("float64").sum()
        )

    if normalized:
        norm_fact = pairlib.scalings.norm_scaling_factor(
            dist_bin_mids, pair_frequencies, anchor=int(1e3)
        )
        pair_frequencies = pair_frequencies / norm_fact
        avg_trans = avg_trans / norm_fact if avg_trans else None

    return (dist_bin_mids[mask], pair_frequencies[mask]), avg_trans


@cli.command("compute-scaling")
@click.argument("pairs_paths", nargs=-1, type=click.Path(exists=True), required=True)
@click.option(
    "--out",
    "-o",
    "out_path",
    required=True,
    type=click.Path(),
    help="The path to the scaling plot output file.",
)
@click.option(
    "--region",
    "-r",
    "region",
    type=str,
    help="UCSC-style coordinates of the genomic region to calculate scalings for.",
)
@click.option(
    "--exclude-chrom",
    "exclude_chroms",
    type=str,
    multiple=True,
    help='Exclude the specified chromosome from the scalings. Optionally add ":left" or ":right" to the argument to only exclude the corresponding arm of the chromosome.',
)
@click.option(
    "--exclude-end-regions",
    "exclude_end_regions",
    type=int,
    default=10000,
    help="Centromeric and telomeric regions of chromosomal arms in bp to exclude from scalings. Default is 10,000.",
)
@click.option(
    "--assembly",
    "-a",
    "assembly",
    type=str,
    nargs=1,
    help="Assembly name to be used for downloading chromsizes.",
)
@click.option(
    "--centromeres",
    "centromeres_path",
    type=click.Path(exists=True),
    help="Path to a text file containing centromere start and end positions. If not provided, a download will be attempted.",
)
@click.option(
    "--normalized",
    "-n",
    is_flag=True,
    help="Normalize the contact frequency up to 1.0.",
)
@click.option(
    "--split-arms",
    is_flag=True,
    default=False,
    help="Plot scalings of left and right chromosomal arms per chromosome, per pairs file. Caching is disabled for this option.",
)
@click.option(
    "--plot-slope",
    is_flag=True,
    default=False,
    help="Plot the slopes of the scaling curves.",
)
@click.option(
    "--label-subplots",
    is_flag=True,
    default=False,
    help="Label subplots as (a) and (b). Disabled by default.",
)
@click.option(
    "--show-average-trans", is_flag=True, help="Show average trans contact frequency."
)
@click.option(
    "--label",
    "-l",
    "labels",
    type=str,
    multiple=True,
    help="One or more labels for the scaling plot curves.",
)
@click.option(
    "--title", "-t", "title", type=str, nargs=1, help="Title text for the scaling plot."
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Do not use cached values. Caching is enabled by default.",
)
def compute_scaling(
    pairs_paths,
    out_path,
    region,
    exclude_chroms,
    exclude_end_regions,
    assembly,
    centromeres_path,
    split_arms,
    normalized,
    plot_slope,
    label_subplots,
    show_average_trans,
    labels,
    title,
    no_cache,
):
    """
    Compute and plot contact frequency vs genomic separation curves for one or more pairs files.
    """
    labels = list(labels)
    # parse left/right arm parameter of chromosomes to exclude
    exclude_chroms = [chrom.split(":") for chrom in exclude_chroms]

    chromsizes = bioframe.fetch_chromsizes(assembly, filter_chroms=False, as_bed=True)
    chromsizes = chromsizes[~chromsizes.chrom.isin(exclude_chroms)]

    if centromeres_path:
        centromeres = {}
        with open(centromeres_path) as file:
            for line in file:
                cols = line.split(" ")
                centromeres[cols[0]] = (int(cols[1]) + int(cols[2])) // 2
    else:
        centromeres = bioframe.fetch_centromeres(assembly)
        centromeres.set_index("chrom", inplace=True)
        centromeres = centromeres.mid.to_dict()

    if len(labels) != 0 and len(pairs_paths) != len(labels) and not split_arms:
        sys.exit("Please provide as many labels as pairs paths.")

    if region:
        regions = bioframe.select(chromsizes, region).reset_index()
    else:
        # use chromosomal arms as separate regions if no regions are specified
        arms = bioframe.split(chromsizes, centromeres)
        # remove user-excluded chromosomes/arms
        for chrom in exclude_chroms:
            if len(chrom) == 1:
                # no arm specified, remove entire chromosome
                arms = arms[arms.chrom != chrom[0]]
            elif chrom[1] == "left":
                # remove specified chromosome with start == 0 (left arm)
                arms = arms[~((arms.chrom == chrom[0]) & (arms.start == 0))]
            elif chrom[1] == "right":
                # remove specified chromosome with start != 0 (right arm)
                arms = arms[~((arms.chrom == chrom[0]) & (arms.start != 0))]

        # remove 40kb from each side (80kb total) of an arm to remove centromere and telomere regions
        arms = bioframe.ops.expand(arms, -exclude_end_regions)
        # remove arms arms with a length of < 0 after removing side regions
        regions = arms[arms.start < arms.end].reset_index()

    all_scalings = []
    all_avg_trans_levels = []

    for idx, path in enumerate(pairs_paths):
        cis_scalings, avg_trans = None, None

        if split_arms:
            # calculate scalings per arm per chromosome
            cis_scalings, trans_levels = pairlib.scalings.compute_scaling(
                path,
                regions,
                chromsizes,
                dist_range=(int(1e1), int(1e9)),
                n_dist_bins=128,
                chunksize=int(1e7),
            )

            # remove unassigned pairs with start/end positions < 0
            cis_scalings = cis_scalings[
                (cis_scalings.start1 > 0)
                & (cis_scalings.end1 > 0)
                & (cis_scalings.start2 > 0)
                & (cis_scalings.end2 > 0)
            ]

            sc_agg = (
                cis_scalings.groupby(["chrom1", "start1", "min_dist", "max_dist"])
                .agg({"n_pairs": "sum", "n_bp2": "sum"})
                .reset_index()
            )
            avail_chroms = set(sc_agg.chrom1)

            for chrom in avail_chroms:
                # calculate scalings for left/right arms (left arms start at position 0 + exclude_end_regions)
                sc_left, avg_trans_left = calc_pair_freqs(
                    sc_agg[
                        (sc_agg.chrom1 == chrom)
                        & (sc_agg.start1 == exclude_end_regions)
                    ],
                    trans_levels,
                    show_average_trans,
                    normalized,
                )
                sc_right, avg_trans_right = calc_pair_freqs(
                    sc_agg[
                        (sc_agg.chrom1 == chrom)
                        & (sc_agg.start1 != exclude_end_regions)
                    ],
                    trans_levels,
                    show_average_trans,
                    normalized,
                )

                dir_path = os.path.join(
                    os.path.dirname(out_path), os.path.basename(path)
                )
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)
                chrom_path = os.path.join(
                    dir_path, "_".join((chrom, os.path.basename(out_path)))
                )
                (
                    plot_scalings(
                        scalings=[sc_left, sc_right],
                        avg_trans_levels=[avg_trans_left, avg_trans_right],
                        plot_slope=plot_slope,
                        labels=["left", "right"],
                        title=chrom,
                        out_path=chrom_path,
                    )
                )
        else:
            if not no_cache:
                # get cached values
                cached = cache.get(path)
                if cached is not None:
                    cis_scalings = (
                        cached["cis_scalings"]
                        if cached["normalized"] == normalized
                        else None
                    )
                    avg_trans = cached["avg_trans"]

            if (
                no_cache
                or cis_scalings is None
                or (avg_trans is None and show_average_trans)
            ):
                print(
                    f"Computing scalings for file {idx + 1}/{len(pairs_paths)} ...",
                    end="\r",
                )
                # caching disabled or no cached values found

                cis_scalings, trans_levels = pairlib.scalings.compute_scaling(
                    path,
                    regions,
                    chromsizes,
                    dist_range=(int(1e1), int(1e9)),
                    n_dist_bins=128,
                    chunksize=int(1e7),
                )
                # remove unassigned pairs with start/end positions < 0
                cis_scalings = cis_scalings[
                    (cis_scalings.start1 >= 0)
                    & (cis_scalings.end1 >= 0)
                    & (cis_scalings.start2 >= 0)
                    & (cis_scalings.end2 >= 0)
                ]

                sc_agg = (
                    cis_scalings.groupby(["min_dist", "max_dist"])
                    .agg({"n_pairs": "sum", "n_bp2": "sum"})
                    .reset_index()
                )

                cis_scalings, avg_trans = calc_pair_freqs(
                    sc_agg, trans_levels, show_average_trans, normalized
                )

                if not no_cache:
                    cache.set(
                        path,
                        {
                            "cis_scalings": cis_scalings,
                            "avg_trans": avg_trans,
                            "normalized": normalized,
                        },
                    )
            else:
                print(
                    f"Retrieved cached values for file {idx + 1}/{len(pairs_paths)}.",
                    end="\r",
                )

            # use file names as labels if labels have not been provided
            labels.append(os.path.basename) if len(labels) < len(pairs_paths) else None

            all_scalings.append(cis_scalings)
            all_avg_trans_levels.append(avg_trans) if avg_trans is not None else None

    if len(all_scalings) > 0 and not split_arms:
        plot_scalings(
            all_scalings,
            all_avg_trans_levels,
            plot_slope,
            label_subplots,
            labels,
            title,
            out_path,
        )


@cli.command("compute-trans-scaling")
@click.argument("cooler_path", nargs=1, type=click.Path(exists=True), required=True)
@click.option(
    "--out",
    "-o",
    "out_path",
    type=click.Path(),
    help="The path to the scaling plot output file.",
)
@click.option(
    "--resolution",
    nargs=1,
    type=int,
    default=1000,
    help="Resolution of the Hi-C data in a .mcool file.",
)
@click.option(
    "--region1",
    "-r1",
    "regions1",
    type=str,
    multiple=True,
    help="The first region of interactions.",
)
@click.option(
    "--region2",
    "-r2",
    "regions2",
    type=str,
    multiple=True,
    help="The second region of interactions.",
)
@click.option(
    "--label",
    "-l",
    "labels",
    type=str,
    multiple=True,
    help="One or more labels for the interaction frequency curves.",
)
@click.option(
    "--title", "-t", "title", type=str, nargs=1, help="Title text for the plot."
)
def compute_trans_scaling(
    cooler_path, out_path, resolution, regions1, regions2, labels, title
):
    chromsizes = bioframe.fetch_chromsizes("sacCer3", filter_chroms=False, as_bed=True)
    avg_contacts = cooltools.expected.diagsum_asymm(
        clr=cooler.Cooler("::/resolutions/".join((cooler_path, str(resolution)))),
        supports1=list(regions1),
        supports2=list(regions2),
        transforms={"balanced": lambda p: p["count"] * p["weight1"] * p["weight2"]},
    )

    avg_contacts["balanced.avg"] = avg_contacts["balanced.sum"] / avg_contacts(
        "n_valid"
    )

    print("...")


@cli.command("clear-cache")
def clear_cache():
    """
    Erase all cached values.
    """
    cache.clear()
    print("Cache cleared.")


if __name__ == "__main__":
    np.seterr(divide="ignore", invalid="ignore")
    cli()
