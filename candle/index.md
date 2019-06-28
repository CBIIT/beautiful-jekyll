---
bigimg: "/img/FNL_ATRF_Pano_4x10.jpg"
title: How to run CANDLE on Biowulf
---
## Sample job (MNIST)

```bash
module load candle/dev
copy_candle_template-new
./submit_candle_job.sh
```

## Installation of supplemental Python environment for KDS

```bash
source /data/BIDS-HPC/public/software/conda/etc/profile.d/conda.sh
conda create -n main3.6 python=3.6
conda activate main3.6
conda install -c conda-forge louvain python-igraph
pip install cmake desc MulticoreTSNE
```

## Modifications for Jurgen's code

### ```jurgen-latest-benchmarking.py```

Take the working code and prepend and append the following snippets (i.e., make it "wrapper compliant"):

*Prepending Python snippet:*

```python
# Run the wrapper_connector script, which (1) appends $SUPP_PYTHONPATH to the Python environment if it's defined and (2) defines the function for loading the hyperparameters
import sys, os
sys.path.append(os.getenv("CANDLE")+'/Supervisor/templates/scripts')
import wrapper_connector
gParameters = wrapper_connector.load_params('params.json')
################ ADD MODEL BELOW USING gParameters DICTIONARY AS CURRENT HYPERPARAMETER SET; DO NOT MODIFY ABOVE #######################################
```

*Appending Python snippet:*

```python
################ ADD MODEL ABOVE USING gParameters DICTIONARY AS CURRENT HYPERPARAMETER SET; DO NOT MODIFY BELOW #######################################
# Ensure that above you DEFINE the history object (as in, e.g., the return value of model.fit()) or val_to_return (a single number) in your model; below we essentially RETURN those values
try: history
except NameError:
    try: val_to_return
    except NameError:
        print("Error: Neither a history object nor a val_to_return variable was defined upon running the model on the current hyperparameter set; exiting")
        exit
    else:
        wrapper_connector.write_history_from_value(val_to_return, 'val_to_return.json')
else:
    wrapper_connector.write_history(history, 'val_to_return.json')
```

Of course, modify the working code to use the hyperparameters by e.g. ```epochs = gParameters['epochs']```.

Then update the following line in submit_candle_job.sh:

```bash
#export MODEL_SCRIPT="$CANDLE/Supervisor/templates/models/wrapper_compliant/mnist_mlp.py" # should be wrapper-compliant
export MODEL_SCRIPT="$(pwd)/jurgen-latest-benchmarking.py" # should be wrapper-compliant
```

### ```scpyParaTest1-benchmarking.txt```

Place the following in scpyParaTest1-benchmarking.txt:

```
[Global_Params]
min_genes=500
min_cells=5
batch_size=256
num_cores=14
use_gpu=True
learning_rate=300
```

Then update the following line in submit_candle_job.sh:

```bash
#export DEFAULT_PARAMS_FILE="$CANDLE/Supervisor/templates/model_params/mnist1.txt"
export DEFAULT_PARAMS_FILE="$(pwd)/scpyParaTest1-benchmarking.txt"
```

### ```upf-benchmarking.txt```

Generate a sample unrolled parameter file using e.g.:

```bash
python /data/BIDS-HPC/public/software/checkouts/fnlcr-bids-hpc/scripts/generate_hyperparameter_grid.py "['batch_size', np.array([2**6,2**7,2**8,2**9,2**10])]" "['num_cores', np.array([1,2,4,8,14])]" "['use_gpu', [false,true]]"  "['learning_rate', np.array([100,150,200,300,500,700,1000])]" > upf-benchmarking.txt
```

Note: Python's False, True, None, should be replaced by JSON's false, true, null.

Then update the following line in submit_candle_job.sh:

```bash
#export WORKFLOW_SETTINGS_FILE="$CANDLE/Supervisor/templates/workflow_settings/upf3.txt"
export WORKFLOW_SETTINGS_FILE="$(pwd)/upf-benchmarking.txt"
```

### Other modifications

The last non-trivial modification is to set the following variable in submit_candle_job.sh:

```bash
export SUPP_PYTHONPATH="/data/BIDS-HPC/public/software/conda/envs/main3.6/lib/python3.6/site-packages"
```
