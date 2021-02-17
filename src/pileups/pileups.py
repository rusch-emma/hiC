import os
import sys
from typing import Tuple

import click
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import bioframe
import cooler
import cooltools
import cooltools.expected
import cooltools.lib.plotting
import cooltools.snipping
from cooltools import snipping


def snip_pileup(clr, resolution, features, chromsizes, flank):
    windows = snipping.make_bin_aligned_windows(
        resolution,
        features['chrom'],
        features['mid'],
        flank_bp=flank
    )
    supports = chromsizes[['chrom', 'start', 'end']].values
    windows = snipping.assign_regions(windows, supports)

    snipper = cooltools.snipping.CoolerSnipper(clr)
    stack = cooltools.snipping.pileup(
        windows,
        snipper.select,
        snipper.snip
    )
    stack = np.nanmean(stack, axis=2)

    return stack


@click.command()
@click.argument(
    'cooler_paths', nargs=-1,
    type=click.Path(exists=True), required=True
)
@click.option(
    '--out', '-o', 'out_path', required=True,
    type=click.Path(),
    help='The path to the pileup plot output file.'
)
@click.option(
    '--resolution', nargs=1,
    type=int, default=1000,
    help='Resolution of the Hi-C data. Default is 1000.'
)
@click.option(
    '--region', '-r', type=str,
    help='Center coordinates of the genomic region to create a pileup of. Not used if --centromeres is provided.'
)
@click.option(
    '--size', '-s', type=int, default=200000,
    help='Size of the pileup window in bp. Default is 200000 bp.'
)
@click.option(
    '--assembly', '-a',
    type=str, nargs=1, required=True,
    help='Assembly name to be used for downloading chromsizes.'
)
@click.option(
    '--exclude-chrom', 'exclude_chroms',
    type=str, multiple=True,
    help='Exclude the specified chromosome from the pileups.'
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
def plot_pileup(
    cooler_paths,
    out_path,
    resolution,
    region,
    size,
    assembly,
    exclude_chroms,
    centromeres_path,
    title
):
    """
    Plots pileups of a specified size around centromeres for an input cooler file. Input two file paths as arguments to create a ratio of pileups instead.
    """
    if len(cooler_paths) > 2:
        sys.exit('Please provide up to 2 cooler files max.')

    clrs = []

    for path in cooler_paths:
        clr_ext = os.path.splitext(path)[1]
        if clr_ext == '.cool':
            clr = cooler.Cooler(path)
        elif clr_ext == '.mcool':
            clr = cooler.Cooler('::/resolutions/'.join((path, str(resolution))))
        else:
            sys.exit('Please provide a .cool or .mcool file.')
        clrs.append(clr)
        
    chromsizes = bioframe.fetch_chromsizes(assembly, filter_chroms=False, as_bed=True)
    chromsizes = chromsizes[~chromsizes.chrom.isin(exclude_chroms)]

    if centromeres_path:
        features = pd.read_csv(centromeres_path, delim_whitespace=True, header=None, names=['chrom', 'start', 'end', 'mid'])
        features['mid'] = features.apply(lambda row: (row['start'] + row['end']) // 2, axis = 1)
    else:
        pass # TODO: implement

    flank = size // 2

    stacks = [snip_pileup(clr, resolution, features, chromsizes, flank) for clr in clrs]

    vmax = -3.75
    vmin = -1.75
    cmap = 'fall'

    if len(stacks) == 2:
        stacks[0] = stacks[0] / stacks[1]
        vmax = 1
        vmin = -1
        cmap = 'RdBu'

    plt.imshow(
        np.log10(stacks[0]),
        vmax=vmax,
        vmin=vmin,
        cmap=cmap
    )
    plt.colorbar(label='log10 mean')
    ticks_px = np.linspace(0, flank * 2 // resolution, 5)
    ticks_kbp = ((ticks_px -ticks_px[-1] / 2) * resolution // 1000).astype(int)
    plt.xticks(ticks_px, ticks_kbp)
    plt.yticks(ticks_px, ticks_kbp)
    plt.xlabel('relative position, kbp')
    plt.ylabel('relative position, kbp')
    plt.title(title)
    plt.savefig(out_path, dpi=300)


if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore')
    plot_pileup()