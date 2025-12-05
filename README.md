# Data Release: IBL Data Download and Processing

This repository contains the data release code associated with the paper:
**VTA dopamine neuron activity produces spatially organized value representations / https://www.biorxiv.org/content/10.1101/2025.11.04.685995v1)**

This script automates the retrieval of electrophysiology, behavior, and video data from the International Brain Laboratory (IBL) database (OpenAlyx). It performs post-processing to align histology to the Allen/Kim atlas, filters spike sorting clusters based on quality metrics, and aligns Reinforcement Learning (RL) model variables.

## ⚠️ Prerequisites

To run this code, you must have the **IBL environment** installed and activated. This ensures all `one-api`, `ibllib`, and `iblatlas` dependencies are met.

If you do not have the IBL environment set up, please refer to the [IBL installation instructions](https://int-brain-lab.github.io/iblenv/installation.html).

### Key Python Dependencies
* `numpy`
* `pandas`
* `scikit-image` (`skimage`)
* `one-api`
* `iblatlas`
* `brainbox`
* `iblutil`

## 📂 Directory Structure & Setup

The script relies on specific local files to perform histology alignment (mapping channels to Allen/Kim atlas regions). **You must create a folder named `histology_files`** in the same directory as the script and populate it with the required atlas files.

Ensure your working directory looks like this:

```text
project_root/
├── download_data.py         # The main script
├── histology_files/         # Required directory
│   ├── allen_subdivisions.tif
│   ├── 41467_2019_13057_MOESM4_ESM.csv
│   └── nonstr.csv
