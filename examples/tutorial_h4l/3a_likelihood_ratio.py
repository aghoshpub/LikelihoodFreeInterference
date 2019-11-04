#!/usr/bin/env python
# coding: utf-8

# # MadMiner particle physics tutorial
# 
# # Part 3a: Training a likelihood ratio estimator
# 
# Johann Brehmer, Felix Kling, Irina Espejo, and Kyle Cranmer 2018-2019

# In part 3a of this tutorial we will finally train a neural network to estimate likelihood ratios. We assume that you have run part 1 and 2a of this tutorial. If, instead of 2a, you have run part 2b, you just have to load a different filename later.

# ## Preparations

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
#get_ipython().magic(u'matplotlib inline')

from madminer.sampling import SampleAugmenter
from madminer import sampling
from madminer.ml import ParameterizedRatioEstimator


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
        logging.getLogger(key).setLevel(logging.WARNING)


# ## 1. Make (unweighted) training and test samples with augmented data

# At this point, we have all the information we need from the simulations. But the data is not quite ready to be used for machine learning. The `madminer.sampling` class `SampleAugmenter` will take care of the remaining book-keeping steps before we can train our estimators:
# 
# First, it unweights the samples, i.e. for a given parameter vector `theta` (or a distribution `p(theta)`) it picks events `x` such that their distribution follows `p(x|theta)`. The selected samples will all come from the event file we have so far, but their frequency is changed -- some events will appear multiple times, some will disappear.
# 
# Second, `SampleAugmenter` calculates all the augmented data ("gold") that is the key to our new inference methods. Depending on the specific technique, these are the joint likelihood ratio and / or the joint score. It saves all these pieces of information for the selected events in a set of numpy files that can easily be used in any machine learning framework.

# In[3]:


sampler = SampleAugmenter('data/delphes_data_shuffled.h5')


# The `SampleAugmenter` class defines five different high-level functions to generate train or test samples:
# - `sample_train_plain()`, which only saves observations x, for instance for histograms or ABC;
# - `sample_train_local()` for methods like SALLY and SALLINO, which will be demonstrated in the second part of the tutorial;
# - `sample_train_density()` for neural density estimation techniques like MAF or SCANDAL;
# - `sample_train_ratio()` for techniques like CARL, ROLR, CASCAL, and RASCAL, when only theta0 is parameterized;
# - `sample_train_more_ratios()` for the same techniques, but with both theta0 and theta1 parameterized;
# - `sample_test()` for the evaluation of any method.
# 
# For the arguments `theta`, `theta0`, or `theta1`, you can (and should!) use the helper functions `benchmark()`, `benchmarks()`, `morphing_point()`, `morphing_points()`, and `random_morphing_points()`, all defined in the `madminer.sampling` module.
# 
# Here we'll train a likelihood ratio estimator with the ALICES method, so we focus on the `extract_samples_train_ratio()` function. We'll sample the numerator hypothesis in the likelihood ratio with 1000 points drawn from a Gaussian prior, and fix the denominator hypothesis to the SM.
# 
# Note the keyword `sample_only_from_closest_benchmark=True`, which makes sure that for each parameter point we only use the events that were originally (in MG) generated from the closest benchmark. This reduces the statistical fluctuations in the outcome quite a bit.

# In[4]:

mpoints = np.array([0,0.5,0.7,0.8,0.9,0.95,0.98,1,1.02,1.05,1.1,1,2,1.5,1.8,2,3,4,4.5,5,5.5,6,7,8,9,10,12,16]) ** 0.25
mpoints = [(t,1) for t in mpoints]
x, theta0, theta1, y, r_xz, t_xz, n_effective = sampler.sample_train_ratio(
    #theta0=sampling.random_morphing_points(500, [('flat', 0., 16.)]),
    theta0=sampling.morphing_points(mpoints),
    theta1=sampling.benchmark('sm'),
    #n_samples=2*10**5, #100000,
    n_samples=2* 10**6,
    folder='./data/samples',
    filename='train_ratio',
    sample_only_from_closest_benchmark=True,
    return_individual_n_effective=True,
)


# For the evaluation we'll need a test sample:

# In[5]:


_ = sampler.sample_test(
    theta=sampling.benchmark('sm'),
    n_samples=1*10**5,
    #n_samples=1*10**6,
    folder='./data/samples',
    filename='test'
)


# You might notice the information about the "eeffective number of samples" in the output. This is defined as `1 / max_events(weights)`; the smaller it is, the bigger the statistical fluctuations from too large weights. Let's plot this over the parameter space:

# In[6]:


#cmin, cmax = 10., 1000.

# cut = (y.flatten()==0)

# fig = plt.figure(figsize=(5,4))

# #sc = plt.scatter(theta0[cut][:,0], theta0[cut][:,1], c=n_effective[cut],
# sc = plt.scatter(np.reshape(theta0[cut], -1), np.reshape(n_effective[cut],-1),
#                  s=60.,
#                  marker='o')
# # plt.scatter(np.reshape(theta0[cut], -1), np.reshape(theta0[cut],-1), c=np.reshape(n_effective[cut],-1),
# #                  s=60., cmap='viridis',
# #                  norm=matplotlib.colors.LogNorm(vmin=cmin, vmax=cmax),
# #                  marker='o')

# #cb = plt.colorbar(sc)
# #cb.set_label('Effective number of samples')

# #plt.xlim(0.,2.)
# #plt.ylim(0.,2.)
# #plt.tight_layout()
# #plt.show()
# plt.savefig("effectiveNSamples.pdf")


# ## 2. Plot cross section over parameter space

# This is not strictly necessary, but we can also plot the cross section as a function of parameter space:

# In[7]:


# thetas_benchmarks, xsecs_benchmarks, xsec_errors_benchmarks = sampler.cross_sections(
#     theta=sampling.benchmarks(list(sampler.benchmarks.keys()))
# )

# thetas_morphing, xsecs_morphing, xsec_errors_morphing = sampler.cross_sections(
#     theta=sampling.random_morphing_points(10, [('flat', 0., 16.)])
# )


# In[8]:


# cmin, cmax = 0., 2.5 * np.mean(xsecs_morphing)

# fig = plt.figure(figsize=(5,4))

# sc = plt.scatter(thetas_morphing[:,0], thetas_morphing[:,1], c=xsecs_morphing,
#             s=40., cmap='viridis', vmin=cmin, vmax=cmax,
#             marker='o')

# plt.scatter(thetas_benchmarks[:,0], thetas_benchmarks[:,1], c=xsecs_benchmarks,
#             s=200., cmap='viridis', vmin=cmin, vmax=cmax, lw=2., edgecolor='black',
#             marker='s')

# cb = plt.colorbar(sc)
# cb.set_label('xsec [pb]')

# plt.xlim(-10.,10.)
# plt.ylim(-10.,10.)
# plt.tight_layout()
# #plt.show()
# plt.savefig("xsec.pdf")


# What  you see here is a morphing algorithm in action. We only asked MadGraph to calculate event weights (differential cross sections, or basically squared matrix elements) at six fixed parameter points (shown here as squares with black edges). But with our knowledge about the structure of the process we can interpolate any observable to any parameter point without loss (except that statistical uncertainties might increase)!

# ## 3. Train likelihood ratio estimator

# It's now time to build the neural network that estimates the likelihood ratio. The central object for this is the `madminer.ml.ParameterizedRatioEstimator` class. It defines functions that train, save, load, and evaluate the estimators.
# 
# In the initialization, the keywords `n_hidden` and `activation` define the architecture of the (fully connected) neural network:

# In[9]:


estimator = ParameterizedRatioEstimator(
    n_hidden=(300,),
    activation="tanh"
)


# To train this model we will minimize the ALICES loss function described in ["Likelihood-free inference with an improved cross-entropy estimator"](https://arxiv.org/abs/1808.00973). Many alternatives, including RASCAL, are described in ["Constraining Effective Field Theories With Machine Learning"](https://arxiv.org/abs/1805.00013) and ["A Guide to Constraining Effective Field Theories With Machine Learning"](https://arxiv.org/abs/1805.00020). There is also SCANDAL introduced in ["Mining gold from implicit models to improve likelihood-free inference"](https://arxiv.org/abs/1805.12244).

# In[ ]:


estimator.train(
    method='alices',
    theta='data/samples/theta0_train_ratio.npy',
    x='data/samples/x_train_ratio.npy',
    y='data/samples/y_train_ratio.npy',
    r_xz='data/samples/r_xz_train_ratio.npy',
    t_xz='data/samples/t_xz_train_ratio.npy',
    alpha=1.,
    n_epochs=20,
)

estimator.save('models/alices')


# ## 4. Evaluate likelihood ratio estimator

# `estimator.evaluate_log_likelihood_ratio(theta,x)` estimated the log likelihood ratio and the score for all combination between the given phase-space points `x` and parameters `theta`. That is, if given 100 events `x` and a grid of 25 `theta` points, it will return 25\*100 estimates for the log likelihood ratio and 25\*100 estimates for the score, both indexed by `[i_theta,i_x]`.

# In[ ]:


# theta_each = np.linspace(-20.,20.,21)
# theta0, theta1 = np.meshgrid(theta_each, theta_each)
# theta_grid = np.vstack((theta0.flatten(), theta1.flatten())).T
# np.save('data/samples/theta_grid.npy', theta_grid)

# theta_denom = np.array([[0.,0.]])
# np.save('data/samples/theta_ref.npy', theta_denom)


# # In[ ]:


# estimator.load('models/alices')

# log_r_hat, _ = estimator.evaluate_log_likelihood_ratio(
#     theta='data/samples/theta_grid.npy',
#     x='data/samples/x_test.npy',
#     evaluate_score=False
# )


# # Let's look at the result:

# # In[ ]:


# bin_size = theta_each[1] - theta_each[0]
# edges = np.linspace(theta_each[0] - bin_size/2, theta_each[-1] + bin_size/2, len(theta_each)+1)

# fig = plt.figure(figsize=(6,5))
# ax = plt.gca()

# expected_llr = np.mean(log_r_hat,axis=1)
# best_fit = theta_grid[np.argmin(-2.*expected_llr)]

# cmin, cmax = np.min(-2*expected_llr), np.max(-2*expected_llr)
    
# pcm = ax.pcolormesh(edges, edges, -2. * expected_llr.reshape((21,21)),
#                     norm=matplotlib.colors.Normalize(vmin=cmin, vmax=cmax),
#                     cmap='viridis_r')
# cbar = fig.colorbar(pcm, ax=ax, extend='both')

# plt.scatter(best_fit[0], best_fit[1], s=80., color='black', marker='*')

# plt.xlabel(r'$\theta_0$')
# plt.ylabel(r'$\theta_1$')
# cbar.set_label(r'$\mathbb{E}_x [ -2\, \log \,\hat{r}(x | \theta, \theta_{SM}) ]$')

# plt.tight_layout()
# #plt.show()
# plt.savefig("result3a.pdf")


# # Note that in this tutorial our sample size was very small, and the network might not really have a chance to converge to the correct likelihood ratio function. So don't worry if you find a minimum that is not at the right point (the SM, i.e. the origin in this plot). Feel free to dial up the event numbers in the run card as well as the training samples and see what happens then!

# # In[ ]:

print("All done..")


