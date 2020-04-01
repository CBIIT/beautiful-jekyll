---
layout: post
date: 2019-12-02
bigimg: "/img/FNL_ATRF_Pano_4x10.jpg"
title: CANDLE on Biowulf
subtitle: "Exercise: Running a Variational Autoencoder Using PyTorch"
---

# Introduction
---

This exercise will get you started running [CANDLE](https://cbiit.github.com/sdsi/candle) on Biowulf. In particular, you will run a [variational autoencoder (VAE)](https://www.google.com/search?q=variational+autoencoder) using [PyTorch](https://pytorch.org), a deep learning library developed by Facebook. A VAE is a type of generative model (capable of "generating" new samples) that has the advantage over the traditional autoencoder that the learned latent space is not discrete and is instead more or less "continuous." The following exercise does not require that you know anything about how VAEs work; you will simply download the example VAE provided by Facebook, make it "CANDLE-compliant," and perform a hyperparameter optimization on it using CANDLE.

The exercise below consists of two parts: (1) ensuring that CANDLE can run the model by testing it first on a single interactive node on Biowulf, and (2) running a full hyperparameter optimization (HPO) on the model with CANDLE by submitting the job to Biowulf in the typical batch mode.

After completing this exercise, you will know exactly how to perform HPO on your own model using CANDLE on Biowulf. Along the way, you will have learned to follow good practices for doing so.

# Links
---

* NIH FAES BIOF 399 presentation PDFs:
  * [George Zaki](mailto:george.zaki@nih.gov)'s presentation: [CANDLE: A Scalable Infrastructure to Accelerate Machine Learning Studies](faes-presentation-nov2019.pdf)
  * [Andrew Weisman](mailto:andrew.weisman@nih.gov)'s presentation: [How to run CANDLE on Biowulf](candle_on_biowulf-faes.pdf)
* CANDLE on Biowulf [homepage](https://cbiit.github.com/sdsi/candle)
* CANDLE on Biowulf [documentation](https://hpc.nih.gov/apps/candle)

# Exercise
---

## Part 1: Ensure, on an interactive SLURM node, that CANDLE can run the model

Create and enter a working directory on your data partition on Biowulf, e.g.,

```bash
mkdir /data/USERNAME/vae_with_pytorch
cd /data/USERNAME/vae_with_pytorch
```

Start an interactive SLURM session on Biowulf:

```bash
sinteractive --gres=gpu:k20x:1 --mem=60G --cpus-per-task=16
```

Once the interactive session has been provisioned, load the ```candle``` module:

```bash
module load candle
```

Import a CANDLE template, which consists of an input file (```.in``` extension) and a "model script" (the script containing the machine/deep learning model you'd like to run). The ```grid``` template is always a solid starting point:

```bash
candle import-template grid
```

Note that if you were to run ```candle submit-job grid_example.in```, you would successfully submit (via SLURM's batch mode) a grid search HPO on the MNIST dataset. As we'd like to start from scratch (as opposed to using an already CANDLE-compliant model script), we can ignore the model script ```mnist_mlp.py``` and will focus on the template CANDLE input file, ```grid_example.in```, which we will shortly modify to be used for the VAE.

Clone the PyTorch examples repository:

```bash
git clone https://github.com/pytorch/examples.git
```

Remember, a prerequisite for running your model script with CANDLE is to make sure your model script runs standalone on Biowulf in the first place. Let's make sure the VAE example you just downloaded works as-is on Biowulf:

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

This line is unnecessary anyway as long as you're following good practice and using an up-to-date version of Python. We followed this good practice by running ```module load python/3.6``` earlier. (By default, CANDLE will always load a recent version of Python.)

Also, ```main.py```'s authors assumed a particular directory structure, whereas we want the script to run exactly where it is without assuming the presence of any other directories (which in this case is a ```data``` directory one level up and a ```results``` directory in the working directory). To address this, add these lines anywhere near the top of the script, e.g., right after the block of ```import``` statements:

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

Note that all we've done so far is modify the model script ```main.py``` to make sure it can be used by CANDLE. While these sorts of changes are representative of common modifications you may need to make to models you find online, they have nothing in particular to do with CANDLE.

Here is a "diff" summary of the changes we have made to ```main.py``` so far:

```diff
@@ -1,4 +1,3 @@
-from __future__ import print_function
 import argparse
 import torch
 import torch.utils.data
@@ -8,6 +8,11 @@ from torchvision import datasets, transforms
 from torchvision.utils import save_image
 
 
+import os
+os.makedirs('data', exist_ok=True)
+os.makedirs('results', exist_ok=True)
+
+
 parser = argparse.ArgumentParser(description='VAE MNIST Example')
 parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                     help='input batch size for training (default: 128)')
@@ -28,11 +33,13 @@ device = torch.device("cuda" if args.cuda else "cpu")
 
 kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
 train_loader = torch.utils.data.DataLoader(
-    datasets.MNIST('../data', train=True, download=True,
+    datasets.MNIST('data', train=True, download=True,
                    transform=transforms.ToTensor()),
     batch_size=args.batch_size, shuffle=True, **kwargs)
 test_loader = torch.utils.data.DataLoader(
-    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
+    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
     batch_size=args.batch_size, shuffle=True, **kwargs)
 
 
@@ -121,12 +128,11 @@ def test(epoch):
     test_loss /= len(test_loader.dataset)
     print('====> Test set loss: {:.4f}'.format(test_loss))
```

The only actual *required* modification to a model script is to return a value (called ```val_to_return```) on which you want to base the hyperparameter optimization, such as the loss calculated by the model on a validation dataset. Let's use the ```test_loss``` variable the model script authors have already defined. Do this by returning ```test_loss``` from the ```test()``` function and then assigning it in the script's body to the variable ```val_to_return```:

```python
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return(test_loss) # add this line to the end of the test() function

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        #test(epoch)
        val_to_return = test(epoch) # assign the return value of test() to the "val_to_return" variable in the main part of the script
        with torch.no_grad():
```

We have not yet defined any hyperparameters within the script, but at least it should now run using CANDLE; let's test that.

Copy the template CANDLE input file you imported earlier into the current directory, renaming it to something more meaningful for this exercise:

```bash
cp ../../grid_example.in vae_with_pytorch.in
```

Open up this input file ```vae_with_pytorch.in``` in a text editor and change the ```model_script``` variable to the model script we'd like to use, ```main.py```, instead of the one in there by default, ```mnist_mlp.py```. (It's best to always use absolute pathnames; that's why the ```$(pwd)/``` in the assignment below.) Since we're using a non-default deep learning library backend (the default is ```keras```), also in the ```&control``` section create a variable called ```dl_backend``` and set its value to ```pytorch```. Also, since we want to just run the test script using CANDLE once in interactive mode (as opposed to running a workflow like HPO), define a ```use_candle``` variable and set its value to ```0```:

```
&control
  model_script="$(pwd)/main.py"
  dl_backend="pytorch"
  use_candle=0
  workflow="grid"
  ngpus=2
  gpu_type="k80"
  walltime="00:20:00"
/
```

The other variables in the ```&control``` section, and the other two sections, don't really matter... again, we just want to make sure CANDLE can run the file once using the currently assigned K20x GPU. (Normally at this point the settings in the ```&default_model``` section of the input file *would* matter, but we didn't yet define these hyperparameters in the model script ```main.py```. We will do this soon!)

Now "submit" the CANDLE input file. Note that by setting ```use_candle=0``` we're telling CANDLE to run the script interactively, as opposed to "submitting" the script to SLURM in batch mode.

```bash
candle submit-job vae_with_pytorch.in
```

While you won't see the output of the script directly in the terminal as before, it is being written to a file called ```subprocess_out_and_err.txt``` in your working directory. (Feel free to log in to the interactive node you've been assigned from Biowulf [e.g., ```ssh cn0605```] and run ```watch nvidia-smi``` to see the GPU running. Or just watch the contents of ```subprocess_out_and_err.txt```.) Whenever you run CANDLE, the output of what you'd expect from running the model outside of CANDLE will always be in a file called ```subprocess_out_and_err.txt```. If you run CANDLE interactively (i.e., on an interactive node using the ```use_candle=0``` setting), this file will be located in your working directory; if you run CANDLE in batch mode (as we will in the second part of this exercise), there will be one file ```subprocess_out_and_err.txt``` in each of the directories in the ```last-exp/run``` folder.

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

Now try running the model script again using CANDLE, checking to see that the default values of the hyperparameters we set in the input file actually have an effect (note in particular that we set ```epochs=2```):

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

Finally, take a look in ```subprocess_out_and_err.txt``` to ensure that the default settings we specified in the ```&default_model``` section of the input file ```vae_with_pytorch.in``` had an effect when CANDLE ran the model script ```main.py```. You should see now that only two epochs have been run!

Here is a "diff" summary of the CANDLE-related changes we made to the model script reinforcing that generally just two simple changes need to be made to your model script in order to make it CANDLE-compliant: (1) specifying the hyperparameters in the ```hyperparams``` dictionary and (2) returning a metric of the performance of a hyperparameter set such as the validation loss in the ```val_to_return``` variable:

```diff
@@ -14,11 +14,14 @@ os.makedirs('results', exist_ok=True)


 parser = argparse.ArgumentParser(description='VAE MNIST Example')
-parser.add_argument('--batch-size', type=int, default=128, metavar='N',
+parser.add_argument('--batch-size', type=int, default=hyperparams['batch_size'], metavar='N',
                     help='input batch size for training (default: 128)')
-parser.add_argument('--epochs', type=int, default=10, metavar='N',
+parser.add_argument('--epochs', type=int, default=hyperparams['epochs'], metavar='N',
                     help='number of epochs to train (default: 10)')
-parser.add_argument('--no-cuda', action='store_true', default=False,
+parser.add_argument('--no-cuda', action='store_true', default=hyperparams['no_cuda'],
                     help='enables CUDA training')
 parser.add_argument('--seed', type=int, default=1, metavar='S',
                     help='random seed (default: 1)')
@@ -120,11 +130,13 @@ def test(epoch):
 
     test_loss /= len(test_loader.dataset)
     print('====> Test set loss: {:.4f}'.format(test_loss))
+    return(test_loss)
 
 if __name__ == "__main__":
     for epoch in range(1, args.epochs + 1):
         train(epoch)
-        test(epoch)
+        val_to_return = test(epoch)
         with torch.no_grad():
             sample = torch.randn(64, 20).to(device)
             sample = model.decode(sample).cpu()
```

A complete, final version of ```main.py``` can be found [here](main.py).

## Part 2: Run HPO on the model using SLURM's batch mode using CANDLE

Now we want to run a full HPO on the VAE model script. We have already defined what the hyperparameters are in the script and what their default values should be; now we just need to define the space of hyperparameters we'd like the HPO to use.

For demonstration purposes, let's vary two out of the three already-defined hyperparameters in a grid search. Run on the command line:

```bash
candle generate-grid "['epochs',np.arange(2,11,2)]" "['batch_size',[64,128,256,512,1024]]"
```

and place the contents of the generated file ```grid_workflow-XXXX.txt``` into the ```&param_space``` section of your input file (delete what's currently there).

In the input file ```vae_with_pytorch.in``` set ```use_candle``` to ```1``` (or delete the setting altogether, as ```1``` is the default value). This tells CANDLE to submit the HPO job defined in the input file to the SLURM scheduler in batch mode, as opposed to running the script just once on the current machine using the default hyperparameter values. Your final input file should look something like this:

```
&control
  model_script="$(pwd)/main.py"
  dl_backend="pytorch"
  use_candle=1
  workflow="grid"
  ngpus=2
  gpu_type="k80"
  walltime="00:20:00"
/

&default_model
  epochs = 2
  batch_size=128
  no_cuda = False
/

&param_space
{"id": "hpset_00001", "epochs": 2, "batch_size": 64}
{"id": "hpset_00002", "epochs": 2, "batch_size": 128}
{"id": "hpset_00003", "epochs": 2, "batch_size": 256}
{"id": "hpset_00004", "epochs": 2, "batch_size": 512}
{"id": "hpset_00005", "epochs": 2, "batch_size": 1024}
{"id": "hpset_00006", "epochs": 4, "batch_size": 64}
{"id": "hpset_00007", "epochs": 4, "batch_size": 128}
{"id": "hpset_00008", "epochs": 4, "batch_size": 256}
{"id": "hpset_00009", "epochs": 4, "batch_size": 512}
{"id": "hpset_00010", "epochs": 4, "batch_size": 1024}
{"id": "hpset_00011", "epochs": 6, "batch_size": 64}
{"id": "hpset_00012", "epochs": 6, "batch_size": 128}
{"id": "hpset_00013", "epochs": 6, "batch_size": 256}
{"id": "hpset_00014", "epochs": 6, "batch_size": 512}
{"id": "hpset_00015", "epochs": 6, "batch_size": 1024}
{"id": "hpset_00016", "epochs": 8, "batch_size": 64}
{"id": "hpset_00017", "epochs": 8, "batch_size": 128}
{"id": "hpset_00018", "epochs": 8, "batch_size": 256}
{"id": "hpset_00019", "epochs": 8, "batch_size": 512}
{"id": "hpset_00020", "epochs": 8, "batch_size": 1024}
{"id": "hpset_00021", "epochs": 10, "batch_size": 64}
{"id": "hpset_00022", "epochs": 10, "batch_size": 128}
{"id": "hpset_00023", "epochs": 10, "batch_size": 256}
{"id": "hpset_00024", "epochs": 10, "batch_size": 512}
{"id": "hpset_00025", "epochs": 10, "batch_size": 1024}
/
```

Note that while we are not varying the third hyperparameter ```no_cuda``` in the hyperparameter space specified in the ```&param_space``` section of the input file, CANDLE knows what value to set it to in the model script because this is specified in the ```&default_model``` section of the input file.

Finally, once again submit the CANDLE job:

```bash
candle submit-job vae_with_pytorch.in
```

Now, once your job picks up by SLURM on Biowulf, a grid search using CANDLE should run, taking about 15 minutes to complete.

The results of the grid search HPO are located in a subdirectory within the ```experiments``` directory. Feel free to tweak the hyperparameter space or anything else and run another CANDLE job; every time you do, another subdirectory will be generated in ```experiments```. As a matter of convenience, CANDLE generates a symbolic link in your working directory called ```last-exp``` to the last experiment (i.e., CANDLE job) that you ran.

The general point of a hyperparameter optimization is to determine the best set of hyperparameters to use in your model, where you've defined some metric of how well each set of hyperparameters performed. In this exercise we are evaluating how the number of epochs and batch size affect how well the variational autoencoder performs on the test set as measured by the loss calculated on the test set. We have specified this loss in CANDLE by assigning it to the ```val_to_return``` variable in the model script.

After the HPO job is complete, in order to easily evaluate how the different sets of hyperparameters affect the calculated test loss in our experiment, we can collect all values of the hyperparameters and test loss into a single CSV file by running:

```bash
candle aggregate-results $(pwd)/last-exp
```

Note that the directory that's the second argument to ```candle```, as usual in CANDLE, must be an absolute path; that's why the ```$(pwd)```.

This produces a file in the working directory called ```candle_results.csv``` and should look something like this:

```
result,dirname,id,epochs,batch_size
105.624,hpset_00022,hpset_00022,10,128
105.705,hpset_00016,hpset_00016,8,64
105.726,hpset_00021,hpset_00021,10,64
106.241,hpset_00017,hpset_00017,8,128
106.731,hpset_00011,hpset_00011,6,64
106.952,hpset_00023,hpset_00023,10,256
107.664,hpset_00012,hpset_00012,6,128
108.353,hpset_00006,hpset_00006,4,64
108.365,hpset_00018,hpset_00018,8,256
109.939,hpset_00007,hpset_00007,4,128
110.033,hpset_00013,hpset_00013,6,256
110.203,hpset_00024,hpset_00024,10,512
112.136,hpset_00019,hpset_00019,8,512
112.386,hpset_00001,hpset_00001,2,64
113.825,hpset_00008,hpset_00008,4,256
115.138,hpset_00014,hpset_00014,6,512
116.039,hpset_00025,hpset_00025,10,1024
116.362,hpset_00002,hpset_00002,2,128
119.596,hpset_00020,hpset_00020,8,1024
121.821,hpset_00009,hpset_00009,4,512
125.091,hpset_00003,hpset_00003,2,256
125.765,hpset_00015,hpset_00015,6,1024
138.186,hpset_00010,hpset_00010,4,1024
141.458,hpset_00004,hpset_00004,2,512
166.273,hpset_00005,hpset_00005,2,1024
```

The ```result``` (first column) is the value specified in your model script by ```val_to_return``` and is followed (after some label columns) by columns of the hyperparameters that you varied, which in this case are ```epochs``` and ```batch_size```. The results are sorted by increasing ```result``` and here show that the best values of ```epochs``` and ```batch_size``` to use (i.e., those that minimize the test loss values in the ```result``` column) are ```epochs~[8,10], batch_size~[64,128]```. In this case, training the VAE for fewer epochs using larger batch sizes produces worse results, i.e., higher losses on the test dataset.

# Summary and next steps
---

You have just taken a cutting-edge deep learning model (a variational autoencoder) straight from a reliable online model repository (PyTorch's own ```examples``` repository) and made a minimal number of modifications to it in order to make it "CANDLE-compliant." You then ran a small hyperparameter optimization (a grid search) on the model using CANDLE on Biowulf in order to determine which set of hyperparameters makes the model perform optimally, which you defined as that which minimizes the loss calculated on the test dataset. To do this you modified a single template input file that you brought into your working directory using ```candle import-template grid```.

While CANDLE is more than just software for hyperparameter optimization (HPO), it excels at this task, which is crucial for developing the best possible model for a particular problem. You now know the exact steps you need to take in order to successfully perform HPO on your own model/data. Feel free to consult the CANDLE on Biowulf [user guide](https://hpc.nih.gov/apps/candle) to learn more. In particular, if you are interested in performing a Bayesian hyperparameter search (as opposed to a grid hyperparameter search), a good place to start is by studying the ```bayesian``` template input file that can be copied over using ```candle import-template bayesian```. Finally, CANDLE is more than capable of running HPO on models written in the ```R``` programming language; try running the example obtained using ```candle import-template r``` in order to see this at work.

Please [contact us](mailto:andrew.weisman@nih.gov) if you have any questions, comments, or suggestions about improving the above exercise or if you need help using CANDLE's HPO functionality for your own needs. We are happy to help! Stay in the know about CANDLE updates on Biowulf by tuning in to the CANDLE [homepage](https://cbiit.github.com/sdsi/candle). In addition, if you are interested in joining a CANDLE-users NIH listserv (we are gauging interest), please let us know as well.