# coding: utf-8

# # MadMiner particle physics tutorial
# 
# # Part 2b: Analyzing events at Delphes level
# 
# Johann Brehmer, Felix Kling, Irina Espejo, and Kyle Cranmer 2018-2019

# In this second part of the tutorial, we'll generate events and extract the observables and weights from them. You have two options: In this notebook we'll do this with Delphes, in the alternative part 2a we stick to parton level.

# ## 0. Preparations

# Before you execute this notebook, make sure you have working installations of MadGraph, Pythia, and Delphes.

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
#get_ipython().magic(u'matplotlib inline')

from madminer.core import MadMiner
from madminer.delphes import DelphesReader
from madminer.sampling import combine_and_shuffle
from madminer.plotting import plot_distributions
import sys
runIteration = int(sys.argv[1])

# In[2]:


# MadMiner output
logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.DEBUG
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)


# Please enter here the path to your MG5 root directory. This notebook assumes that you installed Delphes and Pythia through MG5.

# In[3]:


#mg_dir = '/home/software/MG5_aMC_v2_6_2/'
#mg_dir = '../../../MG5_aMC_v2_6_7/'
mg_dir = '/home/software/MG5_aMC_v2_6_7/'


# ## 1. Generate events

# Let's load our setup:

# In[4]:


miner = MadMiner()
miner.load("data/setup.h5")


# In a next step, MadMiner starts MadGraph and Pythia to generate events and calculate the weights. You can use `run()` or `run_multiple()`; the latter allows to generate different runs with different run cards and optimizing the phase space for different benchmark points. 
# 
# In either case, you have to provide paths to the process card, run card, param card (the entries corresponding to the parameters of interest will be automatically adapted), and an empty reweight card. Log files in the `log_directory` folder collect the MadGraph output and are important for debugging.
# 
# The `sample_benchmark` (or in the case of `run_all`, `sample_benchmarks`) option can be used to specify which benchmark should be used for sampling, i.e. for which benchmark point the phase space is optimized. If you just use one benchmark, reweighting to far-away points in parameter space can lead to large event weights and thus large statistical fluctuations. It is therefore often a good idea to combine at least a few different benchmarks for this option. Here we use the SM and the benchmark "w" that we defined during the setup step.
# 
# One slight annoyance is that MadGraph only supports Python 2. The `run()` and `run_multiple()` commands have a keyword `initial_command` that let you load a virtual environment in which `python` maps to Python 2 (which is what we do below). Alternatively / additionally you can set `python2_override=True`, which calls `python2.7` instead of `python` to start MadGraph.

# In[5]:

# Create new run card
#bashCommand = "cp cards/run_card_signal_h4l.dat temp/run_card_signal_h4l_runIter{}.dat".format(runIteration)
bashCommand = "cp cards/run_card_signal_h4l_tiny.dat temp/run_card_signal_h4l_runIter{}.dat".format(runIteration)
import subprocess
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
print (output)
print (error)

# add unique random seed that's far away from other directories 
#(sucessive processes will automatically get different seed in run_multiple() )
subprocess.call(["sed", "-i", "-e",  "/iseed/s/0/{}/".format(runIteration*20 + 20), "temp/run_card_signal_h4l_runIter{}.dat".format(runIteration)])
# import fileinput
# for line in fileinput.input("temp/run_card_signal_h4l_runIter{}.dat".format(runIteration), inplace=True):
#     # inside this loop the STDOUT will be redirected to the file
#     # the comma after each print statement is needed to avoid double line breaks
#     print line.replace("0    = iseed", "{}    = iseed".format(runIteration*20 + 20)),

# miner.run(
#     sample_benchmark='sm',
#     mg_directory=mg_dir,
#     mg_process_directory='./mg_processes/signal_pythia_runIter{}'.format(runIteration),
#     proc_card_file='cards/proc_card_signal.dat',
#     param_card_template_file='cards/param_card_h4l_WZHModified_MGdefault_template.dat',
#     pythia8_card_file='cards/pythia8_card.dat',
#     run_card_file='cards/run_card_signal_h4l.dat',
#     log_directory='logs/signal',
#     initial_command="source activate python2",
# )


# In[6]:

#benchmarks = ['sm', 'no-higgs','0.5_k','0.8_k','0.9_k', '1.2_k','1.35_k', '1.5_k']; lheDir = 
benchmarks = ['sm',]; lheDir = './mg_processes/signal_pythia_all_runIter{}'.format(runIteration)
#benchmarks = ['sm', 'no-higgs','0.8_k', '1.5_k']; lheDir = './mg_processes/signal_pythia_all_runIter{}'.format(runIteration)
#benchmarks = ['sm', '1.2_k','1.35_k']; lheDir = './mg_processes/signal_pythia_additional_runIter{}'.format(runIteration)
#additional_benchmarks = ['1.2_k','1.35_k'] 

# In[7]:

miner.run_multiple(
    only_prepare_script=True,
    sample_benchmarks=benchmarks,
    mg_directory=mg_dir,
    mg_process_directory=lheDir,
    proc_card_file='cards/proc_card_signal.dat',
    param_card_template_file='cards/param_card_h4l_WZHModified_MGdefault_template.dat',
    pythia8_card_file='cards/pythia8_card.dat',
    #run_card_files=['cards/run_card_signal_h4l.dat'],
    run_card_files=["temp/run_card_signal_h4l_runIter{}.dat".format(runIteration)],
    log_directory='logs/signal',
    #initial_command="conda activate madminer",#"source activate python2",
    #initial_command="source activate python2",
    python2_override=True,
)


# This will take a moment -- time for a coffee break!
# 
# After running any event generation through MadMiner, you should check whether the run succeeded: are the usual output files there (LHE and HepMC), do the log files show any error messages? MadMiner does not (yet) perform any explicit checks, and if something went wrong in the event generation, it will only notice later when trying to load the event files.

# ### Backgrounds

# We can also easily add other processes like backgrounds. An important option is the `is_background` keyword, which should be used for processes that do *not* depend on the parameters theta. `is_background=True` will disable the reweighting and re-use the same weights for all cross sections.
# 
# To reduce the runtime of the notebook, the background part is commented out here. Feel free to activate it and let it run during a lunch break.

# In[8]:


"""
miner.run(
    is_background=True,
    sample_benchmark='sm',
    mg_directory=mg_dir,
    mg_process_directory='./mg_processes/background_pythia',
    proc_card_file='cards/proc_card_background.dat',
    pythia8_card_file='cards/pythia8_card.dat',
    param_card_template_file='cards/param_card_template.dat',
    run_card_file='cards/run_card_background.dat',
    log_directory='logs/background',
)
"""


# Finally, note that both `MadMiner.run()` and `MadMiner.run_multiple()` have a `only_create_script` keyword. If that is set to True, MadMiner will not start the event generation directly, but prepare folders with all the right settings and ready-to-run bash scripts. This might make it much easier to generate Events on a high-performance computing system. 
