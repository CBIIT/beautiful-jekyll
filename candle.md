---
bigimg: "/img/FNL_ATRF_Pano_4x10.jpg"
title: CANDLE on Biowulf
---

# Updates
---

## 12/11/19

* We have made CANDLE easier to use! Now, only a [single input file](https://github.com/fnlcr-bids-sdsi/candle/blob/master/templates/examples/grid/grid_example.in) (instead of three) is required (the old way still works though). Try it out by running:
  ```bash
  module load candle
  candle import-template grid
  candle submit-job grid_example.in
  ```
  The full user manual and examples are located as always at the [CANDLE on Biowulf documentation](https://hpc.nih.gov/apps/candle).
* We have written a [step-by-step exercise](https://cbiit.github.com/sdsi/vae_with_pytorch) for implementing a model (a variational autoencoder) straight from GitHub (PyTorch's official examples repository) into CANDLE. Work through it and you will learn how to implement your own model into CANDLE, as well as the best practices for doing so.

## 8/6/19

* The CANDLE documentation is now live on [HPC@NIH's website](https://hpc.nih.gov/apps/candle)! This is only for the docs; please keep the page you're on bookmarked for updates, FAQ, etc.

# Notices
---

* *8/6/19:* If you are interested in joining a CANDLE-users NIH listserv, please [let us know](mailto:andrew.weisman@nih.gov); we are in the process of gauging interest.

# Links
---

* [CANDLE on Biowulf documentation](https://hpc.nih.gov/apps/candle)
* [Implementing a variational autoencoder using PyTorch into CANDLE](https://cbiit.github.com/sdsi/vae_with_pytorch)