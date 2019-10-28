#!/usr/bin/env python
# coding: utf-8

# # MadMiner particle physics tutorial
# 
# # Part 4a: Limit setting
# 
# Johann Brehmer, Felix Kling, Irina Espejo, and Kyle Cranmer 2018-2019

# In part 4a of this tutorial we will use the networks trained in step 3a and 3b to calculate the expected limits on our theory parameters.

# ## 0. Preparations

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals

import six
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
#get_ipython().magic(u'matplotlib inline')


from madminer.limits import AsymptoticLimits
from madminer.sampling import SampleAugmenter
from madminer import sampling
from madminer.plotting import plot_histograms


# In[2]:


# MadMiner output
logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.INFO
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        # print("Deactivating logging output for", key)
        logging.getLogger(key).setLevel(logging.WARNING)


# ## 1. Preparations

# In the end, what we care about are not plots of the log likelihood ratio, but limits on parameters. But at least under some asymptotic assumptions, these are directly related. MadMiner makes it easy to calculate p-values in the asymptotic limit with the `AsymptoticLimits` class in the `madminer.limits`: 

# In[3]:


#limits = AsymptoticLimits('data/lhe_data_shuffled.h5')
limits = AsymptoticLimits('data/delphes_data_shuffled.h5')


# This class provids two high-level functions:
# - `AsymptoticLimits.observed_limits()` lets us calculate p-values on a parameter grid for some observed events, and
# - `AsymptoticLimits.expected_limits()` lets us calculate expected p-values on a parameter grid based on all data in the MadMiner file.
# 
# First we have to define the parameter grid on which we evaluate the p-values.

# In[4]:


grid_ranges = [(-20., 20.), (-20., 20.)]
grid_resolutions = [25, 25]


# What luminosity (in inverse pb) are we talking about?

# In[5]:


lumi = 10000.


# In[6]:


p_values = {}
mle = {}


# ## 2. Expected limits based on histogram

# First, as a baseline, let us calculate the expected limits based on a simple jet pT histogram. Right now, there are not a lot of option for this function; MadMiner even calculates the binning automatically. (We will add more functionality!)
# 
# The keyword `include_xsec` determines whether we include information from the total rate or just use the shapes. Since we don't model backgrounds and systematics in this tutorial, the rate information is unrealistically large, so we leave it out here.

# In[7]:


theta_grid, p_values_expected_histo, best_fit_expected_histo, _, _, (histos, observed, observed_weights) = limits.expected_limits(
    mode="histo",
    hist_vars=["pt_j1"],
    theta_true=[0.,0.],
    grid_ranges=grid_ranges,
    grid_resolutions=grid_resolutions,
    luminosity=lumi,
    include_xsec=False,
    return_asimov=True
)

p_values["Histogram"] = p_values_expected_histo
mle["Histogram"] = best_fit_expected_histo


# With `mode="rate"`, we could calculate limits based on only the rate -- but again, since the rate is extremely powerful when backgrounds and systematics are not taken into account, we don't do that in this tutorial.

# Let's visualize the likelihood estimated with these histograms:

# In[8]:


indices = [12 + i * 25 for i in [6,9,12,15,18]]

fig = plot_histograms(
    histos=[histos[i] for i in indices],
    observed=[observed[i] for i in indices],
    observed_weights=observed_weights,
    histo_labels=[r"$\theta_0 = {:.2f}$".format(theta_grid[i,0]) for i in indices],
    xlabel="Jet $p_T$",
    xrange=(0.,500.),
)

#plt.show()
plt.savefig("histoPlot.pdf")


# ## 3. Expected limits based on ratio estimators

# Next, `mode="ml"` allows us to calculate limits based on any `ParamterizedRatioEstimator` instance like the ALICES estimator trained above:

# In[ ]:


theta_grid, p_values_expected_alices, best_fit_expected_alices, _, _, _ = limits.expected_limits(
    mode="ml",
    model_file='models/alices',
    theta_true=[0.,0.],
    grid_ranges=grid_ranges,
    grid_resolutions=grid_resolutions,
    luminosity=lumi,
    include_xsec=False,
)

p_values["ALICES"] = p_values_expected_alices
mle["ALICES"] = best_fit_expected_alices


# ## 4. Expected limits based on score estimators

# To get p-values from a SALLY estimator, we have to use histograms of the estimated score:

# In[ ]:


theta_grid, p_values_expected_sally, best_fit_expected_sally, _, _, (histos, observed, observed_weights) = limits.expected_limits(
    mode="sally",
    model_file='models/sally',
    theta_true=[0.,0.],
    grid_ranges=grid_ranges,
    grid_resolutions=grid_resolutions,
    luminosity=lumi,
    include_xsec=False,
    return_asimov=True,
)

p_values["SALLY"] = p_values_expected_sally
mle["SALLY"] = best_fit_expected_sally


# Let's have a look at the underlying 2D histograms:

# In[ ]:


indices = [12 + i * 25 for i in [0,6,12,18,24]]

fig = plot_histograms(
    histos=[histos[i] for i in indices],
    observed=observed[0,:100,:],
    observed_weights=observed_weights[:100],
    histo_labels=[r"$\theta_0 = {:.2f}$".format(theta_grid[i,0]) for i in indices],
    xlabel=r'$\hat{t}_0(x)$',
    ylabel=r'$\hat{t}_1(x)$',
    xrange=(-3.,.5),
    yrange=(-3.,3.),
    log=True,
    zrange=(1.e-3,1.),
    markersize=10.
)
    


# As you see, the binning is the same for every parameter point. This is different for `mode="adaptive-sally"`, which optimizes the binning for every grid point differently:

# In[ ]:


theta_grid, p_values_expected_sally, best_fit_expected_sally, _, _, (histos, observed, observed_weights) = limits.expected_limits(
    mode="adaptive-sally",
    model_file='models/sally',
    theta_true=[0.,0.],
    grid_ranges=grid_ranges,
    grid_resolutions=grid_resolutions,
    luminosity=lumi,
    include_xsec=False,
    return_asimov=True,
)

p_values["SALLY-adaptive"] = p_values_expected_sally
mle["SALLY-adaptive"] = best_fit_expected_sally


# In[ ]:


indices = [162, 168, 234, 306, 318, 462]

fig = plot_histograms(
    histos=[histos[i] for i in indices],
    observed=[observed[i, :100, :] for i in indices],
    observed_weights=observed_weights[:100],
    histo_labels=[r"$\theta = ({:.0f},{:.0f})$".format(theta_grid[i,0], theta_grid[i,1]) for i in indices],
    xlabel=r'$\hat{t}_0(x)$',
    ylabel=r'$\hat{t}_1(x)$',
    xrange=(-2.,2),
    yrange=(-2.,2.),
    log=True,
    zrange=(1.e-3,1.),
    markersize=10.
)
    


# We can also use the "SALLINO" strategy, which constructs 1D histogram of `theta * score` at every parameter point on the grid:

# In[ ]:


theta_grid, p_values_expected_sallino, best_fit_expected_sallino, _, _, histos = limits.expected_limits(
    mode="sallino",
    model_file='models/sally',
    theta_true=[0.,0.],
    grid_ranges=grid_ranges,
    grid_resolutions=grid_resolutions,
    luminosity=lumi,
    include_xsec=False,
)

p_values["SALLINO"] = p_values_expected_sallino
mle["SALLINO"] = best_fit_expected_sallino


# ## 5. Expected limits based on likelihood estimators

# In[ ]:


theta_grid, p_values_expected_scandal, best_fit_expected_scandal, _, _, _ = limits.expected_limits(
    mode="ml",
    model_file='models/scandal',
    theta_true=[0.,0.],
    grid_ranges=grid_ranges,
    grid_resolutions=grid_resolutions,
    luminosity=lumi,
    include_xsec=False,
)

p_values["SCANDAL"] = p_values_expected_scandal
mle["SCANDAL"] = best_fit_expected_scandal


# ## 6. Toy signal

# In addition to these expected limits (based on the SM), let us inject a mock signal. We first generate the data:

# In[ ]:


#sampler = SampleAugmenter('data/lhe_data_shuffled.h5')
sampler = SampleAugmenter('data/delphes_data_shuffled.h5')
sc = 1.#1./16.52
x_observed, _, _ = sampler.sample_test(
    #theta=sampling.morphing_point([5.,1.]),
    theta=sampling.morphing_point([15.2*sc,0.1*sc]),
    n_samples=1000,
    #n_samples=100000,
    folder=None,
    filename=None,
)


# In[ ]:


_, p_values_observed, best_fit_observed, _, _, _ = limits.observed_limits(
    x_observed=x_observed,
    mode="ml",
    model_file='models/alices',
    grid_ranges=grid_ranges,
    grid_resolutions=grid_resolutions,
    luminosity=lumi,
    include_xsec=False,
)

p_values["ALICES signal"] = p_values_observed
mle["ALICES signal"] = best_fit_observed


# ## 7. Plot

# Let's plot the results:

# In[ ]:


show = "SALLY"

bin_size = (grid_ranges[0][1] - grid_ranges[0][0])/(grid_resolutions[0] - 1)
edges = np.linspace(grid_ranges[0][0] - bin_size/2, grid_ranges[0][1] + bin_size/2, grid_resolutions[0] + 1)
centers = np.linspace(grid_ranges[0][0], grid_ranges[0][1], grid_resolutions[0])

fig = plt.figure(figsize=(6,5))
ax = plt.gca()

cmin, cmax = 1.e-2, 1.

pcm = ax.pcolormesh(
    edges, edges, p_values[show].reshape((grid_resolutions[0], grid_resolutions[1])).T,
    norm=matplotlib.colors.LogNorm(vmin=cmin, vmax=cmax),
    cmap='Greys_r'
)
cbar = fig.colorbar(pcm, ax=ax, extend='both')

for i, (label, p_value) in enumerate(six.iteritems(p_values)):
    plt.contour(
        centers, centers, p_value.reshape((grid_resolutions[0], grid_resolutions[1])).T,
        levels=[0.32],
        linestyles='-', colors='C{}'.format(i)
    )
    plt.scatter(
        theta_grid[mle[label]][0], theta_grid[mle[label]][1],
        s=80., color='C{}'.format(i), marker='*',
        label=label
    )

plt.legend()

plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
cbar.set_label('Expected p-value ({})'.format(show))

plt.tight_layout()
#plt.show()
plt.savefig("4aResult.pdf")

# In[ ]:





# In[ ]:

print("All done...")


