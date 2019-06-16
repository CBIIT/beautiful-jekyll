---
bigimg: "/img/FNL_ATRF_Pano_4x10.jpg"
title: How to modify the CANDLE templates
---
The CANDLE templates [submit_candle_job.sh](https://github.com/ECP-CANDLE/Supervisor/blob/fnlcr/templates/submit_candle_job.sh) and [run_without_candle.sh](https://github.com/ECP-CANDLE/Supervisor/blob/fnlcr/templates/run_without_candle.sh) run as-is as examples on the MNIST dataset.  This page will show you how to modify these template scripts for your own use.

## How to modify `submit_candle_job.sh` for your own use

As introduced [here](https://cbiit.github.io/fnlcr-bids-hpc/documentation/candle/how_to_run_candle_on_biowulf), the `submit_candle_job.sh` [template](https://github.com/ECP-CANDLE/Supervisor/blob/fnlcr/templates/submit_candle_job.sh) is the script you run in order to start a CANDLE job (note that it is not a batch submission script).  Here is the code:

```bash
#!/bin/bash

# Always load the candle module
module load candle


#### MODIFY ONLY BELOW ####################################################################
# Load desired Python version or Conda environment
# Load other custom environment settings here
module load python/3.6

# Model specification
export MODEL_PYTHON_DIR="$CANDLE/Supervisor/templates/models/mnist"
export MODEL_PYTHON_SCRIPT="mnist_mlp"
export DEFAULT_PARAMS_FILE="$CANDLE/Supervisor/templates/model_params/mnist1.txt"

# Workflow specification
export WORKFLOW_TYPE="upf"
export WORKFLOW_SETTINGS_FILE="$CANDLE/Supervisor/templates/workflow_settings/upf3.txt"

# Job specification
export EXPERIMENTS="$(pwd)/experiments" # this will contain the job output; ensure this directory exists
export MODEL_NAME="mnist_upf_test"
export OBJ_RETURN="val_loss"

# Scheduler settings
export PROCS="4" # note that PROCS-1 and PROCS-2 are actually used for UPF and mlrMBO computations, respectively
export PPN="1"
export WALLTIME="00:10:00"
export GPU_TYPE="k80" # the choices on Biowulf are p100, k80, v100, k20x
export MEM_PER_NODE="20G"
#### MODIFY ONLY ABOVE ####################################################################


# Call the workflow; DO NOT MODIFY
$CANDLE/Supervisor/workflows/$WORKFLOW_TYPE/swift/workflow.sh $SITE -a $CANDLE/Supervisor/workflows/common/sh/cfg-sys-$SITE.sh $WORKFLOW_SETTINGS_FILE
```

As indicated by the `...####...` lines, the first thing to ensure is that the correct version of Python is loaded (whether via `modules` or using a Conda environment).  In addition, you can optionally add additional modules or environment settings (e.g., Bash variable settings) that your model may require.

Next, here are the variables where changes would most likely be made:

* Model variables:
  * `MODEL_PYTHON_DIR`: Full path of the directory in which your CANDLE-compliant Python script lies
  * `MODEL_PYTHON_SCRIPT`: Name of your CANDLE-compliant Python script, without the `.py` suffix ([examples](https://github.com/ECP-CANDLE/Supervisor/tree/fnlcr/templates/models))
  * `DEFAULT_PARAMS_FILE`: Location of your [default parameters file](https://cbiit.github.io/fnlcr-bids-hpc/documentation/candle/how_to_make_your_code_candle_compliant) ([examples](https://github.com/ECP-CANDLE/Supervisor/tree/fnlcr/templates/model_params))
* Workflow variables:
  * `WORKFLOW_TYPE`: Desired workflow (currently either `upf` or `mlrMBO`)
  * `WORKFLOW_SETTINGS_FILE`: Location of the workflow settings file, such as the unrolled parameter file ([examples](https://github.com/ECP-CANDLE/Supervisor/tree/fnlcr/templates/workflow_settings))
* Job variables:
  * `EXPERIMENTS`: Location to contain the results of all the hyperparameter optimizations (this directory must exist and can be created automatically using the [copy_candle_template](https://cbiit.github.io/fnlcr-bids-hpc/documentation/candle/how_to_run_candle_on_biowulf) script)
  * `MODEL_NAME`: Name of the job (does not affect actual execution)
  * `OBJ_RETURN`: Variable used to guide more intelligent HPO algorithms such as mlrMBO (currently can be set to `val_loss`, `val_corr`, or `val_dice_coef`)
* Scheduler variables:
  * `PROCS`: Number of GPUs to use for the job (actually, PROCS-1 GPUs are used for UPF and PROCS-2 for mlrMBO)
  * `PPN`: GPU processes per node (generally set this to `1`)
  * `WALLTIME`: How long you expect the entire hyperparameter optimization to take (`HH:MM:SS`)
  * `GPU_TYPE`: Type of GPU card to use; from worst to best on Biowulf: `k20x`, `k80`, `p100`, `v100`
  * `MEM_PER_NODE`: Maximum amount of memory expected to be used for a single job (i.e., for a single set of hyperparameters)

## How to modify `run_without_candle.sh` for your own use

As explained [here](https://cbiit.github.io/fnlcr-bids-hpc/documentation/candle/how_to_make_your_code_candle_compliant), the `run_without_candle.sh` [template](https://github.com/ECP-CANDLE/Supervisor/blob/fnlcr/templates/run_without_candle.sh) is primarily used for testing whether you've successfully made your model CANDLE-compliant.  Here is the code:

```bash
#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --mem=20G
#SBATCH --gres=gpu:k80:1
#SBATCH --time=00:05:00
#SBATCH --job-name=mnist_test_no_candle

# Always load the candle module for, e.g., finding the Benchmark class... DO NOT MODIFY
module load candle

# Load desired Python version or Conda environment
# Load other custom environment settings here
module load python/3.6

# Set the file that the Python script below will read in order to determine the model parameters
export DEFAULT_PARAMS_FILE="$CANDLE/Supervisor/templates/model_params/mnist1.txt"

# Run the model
python $CANDLE/Supervisor/templates/models/mnist/mnist_mlp.py
```

Here are the lines where changes would most likely be made:

* `#SBATCH --mem=20G`: Change `20G` to the largest amount of memory your model may require for a single job
* `#SBATCH --gres=gpu:k80:1`: Since we're only testing CANDLE-compliance, only lower-end GPUs, such as the `k20x` or the `k80`, are likely required
* `#SBATCH --time=00:05:00`: Set this to however long you expect your job to run (`HH:MM:SS`)
* `module load python/3.6`: Ensure the correct version of Python is loaded (whether via `modules` or using a Conda environment) and, if your model requires additional modules or environment settings (e.g., Bash variable settings), add them here
* `export DEFAULT_PARAMS_FILE="$CANDLE/Supervisor/templates/model_params/mnist1.txt"`: Set the `$DEFAULT_PARAMS_FILE` variable to the location of your [default parameters file](https://cbiit.github.io/fnlcr-bids-hpc/documentation/candle/how_to_make_your_code_candle_compliant), e.g., `export DEFAULT_PARAMS_FILE="/path/to/params/file.txt"`
* `python $CANDLE/Supervisor/templates/models/mnist/mnist_mlp.py`: Set this to your CANDLE-compliant Python script, e.g., `python /path/to/candle/compliant/script.py`

## Notes

* The `Supervisor` top directory referred to by links above that point to GitHub is present on Biowulf at `$CANDLE/Supervisor` (if you have the `candle` module loaded) or `/data/BIDS-HPC/public/candle/Supervisor`. In the links, just remove "blob/fnlcr/" or "tree/fnlcr/".  E.g., the file described by the link [https://github.com/ECP-CANDLE/Supervisor/blob/fnlcr/templates/model_params/mnist1.txt](https://github.com/ECP-CANDLE/Supervisor/blob/fnlcr/templates/model_params/mnist1.txt) is located on Biowulf at `/data/BIDS-HPC/public/candle/Supervisor/templates/model_params/mnist1.txt`.
* Feel free to email [Andrew Weisman](mailto:andrew.weisman@nih.gov) with any questions.