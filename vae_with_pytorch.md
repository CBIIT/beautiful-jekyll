---
bigimg: "/img/FNL_ATRF_Pano_4x10.jpg"
title: CANDLE on Biowulf
subtitle: "Exercise: Running a Variational Autoencoder using PyTorch"
fontsize: 10pt
---

# Introduction

This exercise will get you started running CANDLE on Biowulf. In particular, you will run a variational autoencoder (VAE) using PyTorch, a deep learning library developed at Facebook. A VAE is a type of generative model (capable of "generating" new samples) that has the advantage over the traditional autoencoder that the learned latent space is not discrete and instead is more or less "continuous." The following exercise does not require that you know anything about how VAEs work; you will simply download the example VAE provided by Facebook, make it "CANDLE-compliant," and perform a hyperparameter optimization on it using CANDLE.

The exercise below consists of two main steps: (1) ensure that CANDLE can run the model by testing it first on a single interactive node on Biowulf, and (2) run a full hyperparameter optimization (HPO) on the model with CANDLE by submitting the job to Biowulf in the typical batch mode.

After completing this exercise, you will know exactly how to perform HPO on your own model using CANDLE on Biowulf.

# Links

* Classroom presentation PDFs:
  * [George Zaki](mailto:george.zaki@nih.gov)'s presentation: [CANDLE: A Scalable Infrastructure to Accelerate Machine Learning Studies](faes-presentation-nov2019.pdf)
  * [Andrew Weisman](mailto:andrew.weisman@nih.gov)'s presentation: [How to run CANDLE on Biowulf](candle_on_biowulf-faes.pdf)
* [CANDLE on Biowulf homepage](https://cbiit.github.com/sdsi/candle)
* [CANDLE on Biowulf documentation](https://hpc.nih.gov/apps/candle)
* [This webpage](https://cbiit.github.com/sdsi/vae_with_pytorch)

# Exercise

## (1) Ensure, on an interactive SLURM node, that CANDLE can run the model

Create and enter a working directory on your data partition on Biowulf, e.g.,

```bash
mkdir /data/USERNAME/vae_with_pytorch
cd /data/USERNAME/vae_with_pytorch
```

Start an interactive SLURM session on Biowulf:

```bash
sinteractive --gres=gpu:k20x:1 --mem=60G --cpus-per-task=16
```

Once the interactive session has been provisioned, load the development version of CANDLE in order to use the latest CANDLE features (these will be ported over to the main candle module this week):

```bash
module load candle/dev
```

**Note:** Once the main CANDLE module is updated, you only need to run ```module load candle``` as usual.

Import a CANDLE template and delete the unnecessary settings files (they still work, but the single input file method is easier) and the MNIST model script. The ```grid``` template is always a solid starting point:

```bash
candle import-template grid
rm -f submit_candle_job.sh mnist_default_params.txt grid_workflow-mnist.txt README.txt mnist_mlp.py
```

Clone Facebook's (they developed PyTorch) PyTorch examples repository:

```bash
git clone https://github.com/pytorch/examples.git
```

Remember, a prerequisite for using your model script with CANDLE is to make sure your model script runs standalone on Biowulf in the first place. Let's make sure the variational autoencoder (VAE) example you just downloaded works as-is on Biowulf:

```bash
cd examples/vae
module load python/3.6 # load Python on Biowulf first
python main.py
```

If all is working, the script should download some sample data and train the VAE on it for 10 epochs. It should take about about two minutes to run.

Now that we've made sure our model script ```main.py``` runs as-is on Biowulf, we can try running it with CANDLE in our interactive SLURM session.

Open up ```main.py``` in a text editor and comment out the first line by prepending a pound sign to it:

```python
#from __future__ import print_function
```

This line is unnecessary anyway as long as you're following good practice and using an up-to-date version of Python. This followed this good practice by running ```module load python/3.6``` earlier to ensure that we didn't use Biowulf's system version of Python, which is an essentially unsupported version, 2.7.

Also, the script authors put the main code in a ```__name__ == "__main__"``` block, which is used for being able to use the script as a library. Since this is unnecessary in our case (we actually want to run the model script over and over directory, as opposed to using the functions contained in it as part of a library), let's comment out the line defining the block and un-indent the contents of the block:

```python
#if __name__ == "__main__":
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    with torch.no_grad():
        sample = torch.randn(64, 20).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                    'results/sample_' + str(epoch) + '.png')
```

The only actual *required* modification to a model script is to return a value on which you want to base the hyperparameter optimization, such as the loss calculated by the model on a validation dataset. Let's use the ```test_loss``` variable the model script authors have already defined. Do this by returning ```test_loss``` from the ```test()``` function and then assigning it in the script's body to the variable ```val_to_return```:

```python
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return(test_loss)

#if __name__ == "__main__":
for epoch in range(1, args.epochs + 1):
    train(epoch)
    val_to_return = test(epoch)
    with torch.no_grad():
```

We have not yet defined any hyperparameters within the script, but at least it should now run using CANDLE; let's test that.

Move the template CANDLE input file you imported earlier into the current directory, renaming it to something more meaningful for this exercise:

```bash
mv ../../grid_example.in vae_with_pytorch.in
```

Open up this file in a text editor and change the ```model_script``` variable to the model script we'd like to use, ```main.py```. Since we're using a non-default deep learning library backend (the default is ```keras```), create a variable called ```dl_backend``` and set its value to ```pytorch```. Also, since we want to just run the test script using CANDLE once in interactive mode (as opposed to running a workflow), define a ```use_candle``` variable in the ```&control``` section and set its value to ```0```:

```
&control
  model_script="$(pwd)/main.py"
  workflow="grid"
  ngpus=2
  gpu_type="k80"
  walltime="00:20:00"
  dl_backend="pytorch"
  use_candle=0
/
```

For the time being, we need to refrain from having spaces around the equals sign in the ```&control``` section of the input file, just as in a Bash script. We will remove this requirement in the near future.

The other variables in the ```&control``` section, and the other two sections, don't really matter... again, we just want to make sure CANDLE can run the file once on using the currently assigned K20x GPU.

Now "submit" the CANDLE input file. Note that by setting ```use_candle=0``` we're telling CANDLE to run the script interactively, as opposed to "submitting" the script to SLURM in batch mode:

```bash
candle submit-job vae_with_pytorch.in
```

While you won't see the output of the script directly in the terminal like in before, it is being written to a file called ```subprocess_out_and_err.txt```. (Feel free to log into the interactive node you've been assigned from Biowulf [e.g., ```ssh cn0605```] and run ```watch nvidia-smi```) to see the GPU running. Or just watch the contents of ```subprocess_out_and_err.txt```.)

The overall output to the terminal should look something like this:

```
weismanal@cn0605:~/notebook/2019-12-02/prep_for_faes_presentation/examples/vae $ candle submit-job vae_with_pytorch.in
Submitting the CANDLE input file "vae_with_pytorch.in"... 
[-] Unloading python 3.6  ... 
[+] Loading python 3.6  ... 
Loaded pytorch backend
Importing candle utils for pytorch
Configuration file:  /home/weismanal/notebook/2019-12-02/prep_for_faes_presentation/examples/vae/default_params-GENERATED.txt
{'activation': 'relu',
 'batch_size': 128,
 'epochs': 20,
 'num_filters': 32,
 'optimizer': 'rmsprop'}
Params:
{'activation': 'relu',
 'batch_size': 128,
 'datatype': <class 'numpy.float32'>,
 'epochs': 20,
 'experiment_id': 'EXP000',
 'gpus': [],
 'logfile': None,
 'num_filters': 32,
 'optimizer': 'rmsprop',
 'output_dir': '/gpfs/gsfs10/users/weismanal/notebook/2019-12-02/prep_for_faes_presentation/examples/vae/Output/EXP000/RUN000',
 'profiling': False,
 'rng_seed': 7102,
 'run_id': 'RUN000',
 'shuffle': False,
 'timeout': -1,
 'train_bool': True,
 'verbose': None}
Starting run of model_wrapper.sh from candle_compliant_wrapper.py...
Finished run of model_wrapper.sh from candle_compliant_wrapper.py
done
```

Finally, let's set define some hyperparameters in the model script. The general idea is to replace anything you want to modify during a hyperparameter optimization with a dictionary called ```hyperparams```, defining the hyperparameter name using the dictionary "key." For example, let's make the first three possible script arguments hyperparameters:

```python
parser = argparse.ArgumentParser(description='VAE MNIST Example')
#parser.add_argument('--batch-size', type=int, default=128, metavar='N',
parser.add_argument('--batch-size', type=int, default=hyperparams['batch_size'], metavar='N',
                    help='input batch size for training (default: 128)')
#parser.add_argument('--epochs', type=int, default=10, metavar='N',
parser.add_argument('--epochs', type=int, default=hyperparams['epochs'], metavar='N',
                    help='number of epochs to train (default: 10)')
#parser.add_argument('--no-cuda', action='store_true', default=False,
parser.add_argument('--no-cuda', action='store_true', default=hyperparams['no_cuda'],
                    help='enables CUDA training')
```

In the ```&default_model``` section of the CANDLE input file, set the default values of these hyperparameters:

```
&default_model
  epochs = 2
  batch_size=128
  no_cuda = False
/
```

Now try running the script again using CANDLE, checking to see that the default values of the hyperparameters we set in the input file actually have an effect (note that we set ```epochs=2```):

```bash
candle submit-job vae_with_pytorch.in
```

Now the job should run more quickly and the final output to the terminal should look something like:

```
weismanal@cn0605:~/notebook/2019-12-02/prep_for_faes_presentation/examples/vae $ candle submit-job vae_with_pytorch.in
Submitting the CANDLE input file "vae_with_pytorch.in"... 
[-] Unloading python 3.6  ... 
[+] Loading python 3.6  ... 
Loaded pytorch backend
Importing candle utils for pytorch
Configuration file:  /home/weismanal/notebook/2019-12-02/prep_for_faes_presentation/examples/vae/default_params-GENERATED.txt
{'batch_size': 128, 'epochs': 2, 'no_cuda': False}
Params:
{'batch_size': 128,
 'datatype': <class 'numpy.float32'>,
 'epochs': 2,
 'experiment_id': 'EXP000',
 'gpus': [],
 'logfile': None,
 'no_cuda': False,
 'output_dir': '/gpfs/gsfs10/users/weismanal/notebook/2019-12-02/prep_for_faes_presentation/examples/vae/Output/EXP000/RUN000',
 'profiling': False,
 'rng_seed': 7102,
 'run_id': 'RUN000',
 'shuffle': False,
 'timeout': -1,
 'train_bool': True,
 'verbose': None}
Starting run of model_wrapper.sh from candle_compliant_wrapper.py...
Finished run of model_wrapper.sh from candle_compliant_wrapper.py
done
```

Now take a look in ```subprocess_out_and_err.txt``` to ensure that our default settings had an effect. You should see now that only two epochs total have been run!

## (2) Run HPO on the model using SLURM's batch mode using CANDLE

Since it can take a while for jobs to pick up on Biowulf, you can try running a full hyperparamter optimization using CANDLE at home.

Run on the command line:

```bash
candle generate-grid "['epochs',np.arange(2,11,2)]" "['batch_size',[64,128,256,512,1024]]"
```

and place the contents of the generated file ```grid_workflow-XXXX.txt``` into the ```&param_space``` section of your input file (delete what's already there).

In ```vae_with_pytorch.in``` set ```use_candle``` to ```1``` (or delete the setting altogether).

Finally, ```main.py```'s authors assumed a particular directory structure, whereas we want the script to run wherever it's located. To address this, add these lines anywhere near the top of the script, e.g., right after the block of ```import``` statements:

```python
import os
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)
```

Also, remove the ```../``` from the two lines containing ```datasets.MNIST('../data', ```, e.g.,

```python
train_loader = torch.utils.data.DataLoader(
    #datasets.MNIST('../data', train=True, download=True,
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    #datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
```

Finally, once again run:

```bash
candle submit-job vae_with_pytorch.in
```

Now, once your job picks up by SLURM on Biowulf, a grid search using CANDLE should run!