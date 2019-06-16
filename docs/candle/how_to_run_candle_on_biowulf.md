---
bigimg: "/img/FNL_ATRF_Pano_4x10.jpg"
title: How to run CANDLE on Biowulf
---
This page will (1) show you how to run a sample job using CANDLE and (2) indicate what needs to be modified in order to run your own job using CANDLE.

## Step (1): Set up your environment

First, enter some working directory on Biowulf and load the CANDLE environment using `module load candle`.  For example, if your working directory is `/data/USERNAME/candle-test`, run this on the Biowulf command line:

```bash
cd /data/USERNAME/candle-test
module load candle
```

## Step (2): Copy the template submission script to the working directory

Copy the template CANDLE submission script `submit_candle_job.sh` to, and create an empty `experiments` directory in, the working directory by running on Biowulf:

```bash
copy_candle_template
```

## Step (3) (Optional): Modify the variables in the submission script

[Modify the template submission script](https://cbiit.github.io/fnlcr-bids-hpc/documentation/candle/how_to_modify_the_candle_templates) `submit_candle_job.sh` to your needs or leave it unmodified for an example run on the MNIST dataset.

## Step (4): Run the job

Submit the job by running:

```bash
./submit_candle_job.sh
```

(No, there really is no need for "sbatch".)

## Notes

* Feel free to email [Andrew Weisman](mailto:andrew.weisman@nih.gov) with any questions.