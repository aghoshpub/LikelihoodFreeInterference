# coding: utf-8

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
#mg_dir = '../../../MG5_aMC_v2_6_2/'
#mg_dir = '../../../MG5_aMC_v2_6_7/'
mg_dir = '/home/software/MG5_aMC_v2_6_7/'


# ## 1. Generate events

# Let's load our setup:

# In[4]:


miner = MadMiner()
miner.load("data/setup.h5")

#benchmarks = ['sm', 'no-higgs','0.5_k','0.8_k','0.9_k', '1.2_k','1.35_k', '1.5_k']; lheDir = 
benchmarks = ['sm',]; lheDir = './mg_processes/signal_pythia_all_runIter{}'.format(runIteration)
#benchmarks = ['sm', 'no-higgs','0.8_k', '1.5_k']; lheDir = './mg_processes/signal_pythia_all_runIter{}'.format(runIteration)
#benchmarks = ['sm', '1.2_k','1.35_k']; lheDir = './mg_processes/signal_pythia_additional_runIter{}'.format(runIteration)
#additional_benchmarks = ['1.2_k','1.35_k'] 


delphes = DelphesReader('data/setup.h5')


# After creating the `DelphesReader` object, one can add a number of event samples (the output of running MadGraph and Pythia in step 1 above) with the `add_sample()` function.
# 
# In addition, you have to provide the information which sample was generated from which benchmark with the `sampled_from_benchmark` keyword, and set `is_background=True` for all background samples.

# In[10]:


# delphes.add_sample(
#     lhe_filename='mg_processes/signal_pythia_runIter{}/Events/run_01/unweighted_events.lhe.gz'.format(runIteration),
#     hepmc_filename='mg_processes/signal_pythia_runIter{}/Events/run_01/tag_1_pythia8_events.hepmc.gz'.format(runIteration),
#     sampled_from_benchmark='sm',
#     is_background=False,
#     #k_factor=1.1,
# )

# for i, benchmark in enumerate(additional_benchmarks):
#     delphes.add_sample(
#         lhe_filename='mg_processes/signal_pythia2_runIter{}/Events/run_0{}/unweighted_events.lhe.gz'.format(runIteration, i+1),
#         hepmc_filename='mg_processes/signal_pythia2_runIter{}/Events/run_0{}/tag_1_pythia8_events.hepmc.gz'.format(runIteration, i+1),
#         sampled_from_benchmark=benchmark,
#         is_background=False,
#         #k_factor=1.1,
#     )

for i, benchmark in enumerate(benchmarks):
    delphes.add_sample(
        lhe_filename=lheDir + '/Events/run_0{}/unweighted_events.lhe.gz'.format(i+1),
        hepmc_filename=lheDir + '/Events/run_0{}/tag_1_pythia8_events.hepmc.gz'.format(i+1),
        sampled_from_benchmark=benchmark,
        is_background=False,
        #k_factor=1.1,
    )

"""
delphes.add_sample(
    lhe_filename='mg_processes/background_pythia/Events/run_01/unweighted_events.lhe.gz',
    hepmc_filename='mg_processes/background_pythia/Events/run_01/tag_1_pythia8_events.hepmc.gz',
    sampled_from_benchmark='sm',
    is_background=True,
    k_factor=1.0,
"""


# Now we run Delphes on these samples (you can also do this externally and then add the keyword `delphes_filename` when calling `DelphesReader.add_sample()`):

# In[11]:


delphes.run_delphes(
    delphes_directory=mg_dir + '/Delphes',
    delphes_card='cards/delphes_card.dat',
    log_file='logs/delphes.log',
)


# ## 3. Observables and cuts

# The next step is the definition of observables, either through a Python function or an expression that can be evaluated. Here we demonstrate the latter, which is implemented in `add_observable()`. In the expression string, you can use the terms `j[i]`, `e[i]`, `mu[i]`, `a[i]`, `met`, where the indices `i` refer to a ordering by the transverse momentum. In addition, you can use `p[i]`, which denotes the `i`-th particle in the order given in the LHE sample (which is the order in which the final-state particles where defined in MadGraph).
# 
# All of these represent objects inheriting from scikit-hep [LorentzVectors](http://scikit-hep.org/api/math.html#vector-classes), see the link for a documentation of their properties. In addition, they have `charge` and `pdg_id` properties.
# 
# `add_observable()` has an optional keyword `required`. If `required=True`, we will only keep events where the observable can be parsed, i.e. all involved particles have been detected. If `required=False`, un-parseable observables will be filled with the value of another keyword `default`.
# 
# In a realistic project, you would want to add a large number of observables that capture all information in your events. Here we will just define two observables, the transverse momentum of the leading (= higher-pT) jet, and the azimuthal angle between the two leading jets.

# In[12]:


# delphes.add_observable(
#     'pt_j1',
#     'j[0].pt',
#     required=False,
#     default=0.,
# )
# delphes.add_observable(
#     'pt_j2',
#     'j[1].pt',
#     required=False,
#     default=0.,
# )
delphes.add_observable(
    'delta_phi_jj',
    'j[0].deltaphi(j[1]) * (-1. + 2.*float(j[0].eta > j[1].eta))',
    required=False,
    default=-1.,
)
delphes.add_observable(
    'delta_eta_jj',
    'j[0].deltaeta(j[1]) * (-1. + 2.*float(j[0].eta > j[1].eta))',
    required=False,
    default=-1.,
)
delphes.add_observable(
    'invmass_jj',
    '(j[0] + j[1]).m',
    required=False,
    default=-1.,
)
# delphes.add_observable(
#     'eta_j1',
#     'j[0].eta',
#     required=False,
#     default=0.,
# )
# delphes.add_observable(
#     'eta_j2',
#     'j[1].eta',
#     required=False,
#     default=0.,
# )
# delphes.add_observable(
#     'phi_j1',
#     'j[0].phi()',
#     required=False,
#     default=0.,
# )
# delphes.add_observable(
#     'phi_j2',
#     'j[1].phi()',
#     required=False,
#     default=0.,
# )
# delphes.add_observable(
#     'm_j1',
#     'j[0].m',
#     required=False,
#     default=0.,
# )
# delphes.add_observable(
#     'm_j2',
#     'j[1].m',
#     required=False,
#     default=0.,
# )
# delphes.add_observable(
#     'met',
#     'met.pt',
#     required=True,
# )
delphes.add_observable(
    'm4l',
    '(l[0] + l[1] + l[2] + l[3]).m',
    required=True,
)
# delphes.add_observable(
#     'm_l1',
#     'l[0].m',
#     required=True,
# )
# delphes.add_observable(
#     'm_l2',
#     'l[1].m',
#     required=True,
# )
# delphes.add_observable(
#     'm_l3',
#     'l[2].m',
#     required=True,
# )
# delphes.add_observable(
#     'm_l4',
#     'l[3].m',
#     required=True,
# )
# delphes.add_observable(
#     'pt_l1',
#     'l[0].pt',
#     required=True,
# )
# delphes.add_observable(
#     'pt_l2',
#     'l[1].pt',
#     required=True,
# )
# delphes.add_observable(
#     'pt_l3',
#     'l[2].pt',
#     required=True,
# )
# delphes.add_observable(
#     'pt_l4',
#     'l[3].pt',
#     required=True,
# )
# delphes.add_observable(
#     'eta_l1',
#     'l[0].eta',
#     required=True,
# )
# delphes.add_observable(
#     'eta_l2',
#     'l[1].eta',
#     required=True,
# )
# delphes.add_observable(
#     'eta_l3',
#     'l[2].eta',
#     required=True,
# )
# delphes.add_observable(
#     'eta_l4',
#     'l[3].eta',
#     required=True,
# )
# delphes.add_observable(
#     'phi_l1',
#     'l[0].phi()',
#     required=True,
# )
# delphes.add_observable(
#     'phi_l2',
#     'l[1].phi()',
#     required=True,
# )
# delphes.add_observable(
#     'phi_l3',
#     'l[2].phi()',
#     required=True,
# )
# delphes.add_observable(
#     'phi_l4',
#     'l[3].phi()',
#     required=True,
# )
delphes.add_default_observables(n_leptons_max=4, n_photons_max=0, n_jets_max=2, include_met=True, include_visible_sum=True, include_numbers=True, include_charge=True)


# We can also add cuts, again in parse-able strings. In addition to the objects discussed above, they can contain the observables:

# In[13]:


##delphes.add_cut('pt_j1 > 20.')


# ## 4. Analyse events and store data

# The function `analyse_samples` then calculates all observables from the Delphes file(s) generated before and checks which events pass the cuts:

# In[14]:


delphes.analyse_delphes_samples()


# The values of the observables and the weights are then saved in the HDF5 file. It is possible to overwrite the same file, or to leave the original file intact and save all the data into a new file as follows:

# In[15]:


delphes.save('data/delphes_data{}.h5'.format(runIteration))


# ## 5. Plot distributions

# Let's see what our MC run produced:

# In[16]:


# _ = plot_distributions(
#     filename='data/delphes_data.h5',
#     parameter_points=['sm', '5sq-higgs'],
#     line_labels=['SM', 'BSM'],
#     uncertainties='none',
#     n_bins=20,
#     n_cols=3,
#     normalize=True,
# )


# ## 6. Combine and shuffle different samples

# To reduce disk usage, you can generate several small event samples with the steps given above, and combine them now. Note that (for now) it is essential that all of them are generated with the same setup, including the same benchmark points / morphing basis!
# 
# This is generally good practice even if you use just one sample, since the events might have some inherent ordering (e.g. from sampling from different hypotheses). Later when we split the events into a training and test fraction, such an ordering could cause problems.

# In[17]:


# combine_and_shuffle(
#     ['data/delphes_data.h5'],
#     'data/delphes_data_shuffled.h5'
# )

