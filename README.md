### CLI tools and notebooks for downstream analysis of Hi-C data such as creating scaling plots or calculating insulation scores.

* [`src/insulation/`](https://github.com/rusch-emma/hiC/blob/master/src/insulation)
  * [`insulation.py`](https://github.com/rusch-emma/hiC/blob/master/src/insulation/insulation.py): CLI tool for computing insulation scores and plot boundary strengths as a histogram.
* [`src/pileups/`](https://github.com/rusch-emma/hiC/tree/master/src/pileups)
  * [`notebooks/pileups.ipynb`](https://github.com/rusch-emma/hiC/blob/master/src/pileups/notebooks/pileups.ipynb): Jupyter notebook for creating pileups plots around centromeres.
  * [`pileups.py`](https://github.com/rusch-emma/hiC/blob/master/src/pileups/pileups.py): CLI tool for creating pileups plots of a specified size around centromeres.
* [`src/scalings/`](https://github.com/rusch-emma/hiC/tree/master/src/scalings)
  * [`notebooks/scalings.ipynb`](https://github.com/rusch-emma/hiC/blob/master/src/scalings/notebooks/scalings.ipynb): Jupyter notebook for creating scalings plots (contact frequency vs genomic separation).
  * [`notebooks/scalings_arms.ipynb`](https://github.com/rusch-emma/hiC/blob/master/src/scalings/notebooks/scalings_arms.ipynb): Jupyter notebook for creating scaling plots of separate chromosomal arms.
  * [`notebooks/scalings_asym.ipynb`](https://github.com/rusch-emma/hiC/blob/master/src/scalings/notebooks/scalings_asym.ipynb): Jupyter notebook for creating off-diagonal scaling plots.
  * [`scalings.py`](https://github.com/rusch-emma/hiC/blob/master/src/scalings/scalings.py): CLI tool for creating scaling plots for one or more `.pairs` files.
 
 #### Hi-C specific libraries
 * [bioframe](https://github.com/open2c/bioframe)
 * [cooler](https://github.com/open2c/cooler)
 * [cooltools](https://github.com/open2c/cooltools)
 * [pairlib](https://github.com/open2c/pairlib)
 * [pairtools](https://github.com/open2c/pairtools)
