#!/usr/bin/python
import os
import sys
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
# ------------------------------------------------------------------
# Use LLR method to reconstruct the distribution
#   exp[-x^2 / 16^2]

# For exp[-alpha * x^2], Kurt's Sign conference talk states the exact
#   a_i = -2 * alpha * x_i with x_i = i * delta_x
# So with alpha = 1 / 16^2 we should have a_i = -x_i / 128
# Let's try the initial guess a_i = -0.01 * (x_i + delta / 2)
# and hope for convergence to the expected value...
# (The negative sign may be pulled out...)

# Algorithm 1 in section 2.4 of arXiv:1509.08391 lists seven input parameters
# 1) As for the naive case, do N_B = 500 resamples in bootstrap procedure
# 2) No thermalization should be needed (N_{TH} = 0)
#    since we're just pulling numbers from a standard gaussian distribution.
# 3) Let's read in N_{RM}, the number of recursion steps to do,
#    over which the distribution evolves from the initial guess for a_i
# 4) I believe 'E' in this context is just x, making E_min = 0
# 5) Let's see how long it will take to get to E_max = 60
# 6) Following Kurt's Sign conference talk, let's take delta_E = 0.5
#    This implies num_E = (60 - 0) / 0.5 = 120 bins
# 7) I'm not sure about N_{SW}, so let's read that in, too
#    This should be the number of E in [E_i, E_i + delta_E) to generate
#    using the a_i^(n) for the nth recursion step
#    Presumably we won't need O(100k) to reach large E_i with this method...
# For 4d compact U(1), arXiv:1509.08391 uses N_{SW} = 200 and N_{RM} = 400

# !!! Currently only have matplotlib v1.5.2 for python3
# !!! So stick with python2 (and matplotlib v2.2.3) for now

# Parse arguments: random seed and N_{RM}
if len(sys.argv) < 4:
  print("Usage:", str(sys.argv[0]), "<RNG seed> <RM steps> <sweeps>")
  sys.exit(1)
seed = int(sys.argv[1])
N_RM = int(sys.argv[2])
N_SW = int(sys.argv[3])
runtime = -time.time()

# Seed (Mersenne Twister) random number generator
# Use thread-safe RandomState instead of (global) seed
# Former should allow multiple independent streams if desired
prng = np.random.RandomState(seed)

# Hardcoded input parameters discussed above
N_B = 500
E_min = 0.0
E_max = 60.0
delta_E = 0.5
num_E = 120
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# Following Algorithm 1 in section 2.4 of arXiv:1509.08391)
dat = np.empty((num_E, N_B), dtype = np.float)
for B in range(N_B):        # Samples for bootstrapping
  for i in range(num_E):    # Scanning energy range [0, E_max) by delta_E
    E_i = i * delta_E + 10.0
    E_p = E_i + delta_E
    a_i = 0.01 * (E_i + 0.5 * delta_E)      # Initial guess discussed above
    for n in range(N_RM):   # Robbins--Monro recursion
      # Generate N_SW values of E distributed as exp[-a_i * E]
      # We also need E_i <= E < E_p = E_i + delta_E
      # According to the following, this should correspond to
      #   E = -log(u) / a_i
      # with uniform u in the range (exp[-a_i * E_p], exp[-a_i * E_-]]
      # en.wikipedia.org/wiki/Inverse_transform_sampling
      lo = np.exp(-a_i * E_p)
      hi = np.exp(-a_i * E_i)
#      if n == 0:
#        print "%d, %.4g, %.4g, %.4g, %.4g" % (i, E_i, E_p, lo, hi)
      u = prng.uniform(lo, hi, N_SW)
      energies = -np.log(u) / a_i

      # Compute <<delta_E>>_i(a_i^{(n)})
      vev_delta_E = sum(energies) / N_SW - E_i - 0.5 * delta_E

      # Update a_i and check for convergence to -x_i / 128
      a_i = a_i + 12.0 * vev_delta_E / ((n + 1) * delta_E * delta_E)
      print "%d %.4g %.4g" % (n, a_i, -E_i / 128)

    # Recursion complete, so now we have the Bth sample of dat[i]
    dat[i][B] = a_i
    sys.exit(0)
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# Now go through each the energy range [0, E_max) and bootstrap each delta_E
# Do N_B resamples each selecting N_B points with replacement
# References for bootstrapping:
# www.physics.utah.edu/~detar/phys6720/handouts/ftspect/ftspect/node7.html
# --> The uncertainty is the standard deviation of the distribution
# www.itl.nist.gov/div898/handbook/eda/section3/bootplot.htm
sample = np.empty(N_B, dtype = np.float)  # N_B numbers to be pulled from dat
counts = np.empty((N_B, num_E), dtype = np.float)   # As in naive.py
for i in range(num_E):
  for B in range(N_B):
    # Checked that this covers the full range, with replacement
    indices = prng.randint(0, high=N_B, size=N_B)
    for j in range(N_B):
      sample = dat[i][indices[j]]

    counts[i][B] = sample.mean()

# Now average the counts in each bin
# The error is the standard deviation of the population, not of the mean
mean = counts.mean(0)
err = counts.std(0)
print len(mean), len(err)    # 51, 51
print mean
print err
sys.exit(0)

# Set up relative error --- some shenanigans needed to skip NaNs
xList = []
devList = []                # Relative deviation of results from exact
relList = []
for i in range(num_E):   # Try to work around divide-by-zero warnings
  if mean[i] > 0.0:
    xList.append(centers[i])
    devList.append(mean[i] / np.exp(-centers[i]**2 / stdev**2) - 1.0)
    relList.append(err[i] / mean[i])
    # Check absolute deviations in units of bootstrap errors
#    dev = (mean[i] - np.exp(-centers[i]**2 / stdev**2)) / err[i]
#    print "%.2g %.4g" % (centers[i], dev)
x = np.array(xList)
dev = np.array(devList)
relErr = np.array(relList)
#print len(x), len(dev), len(relErr)
#print dev
#print relErr
#sys.exit(0)
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# Set up plots of both distribution and relative error
# Create an instance of plt with two subplots
# (MNi) denotes the ith plot in an MxN array
# Put distribution on top
# Aim for two roughly 600x300 plots on top of each other
fig = plt.figure(figsize=(6, 7))
dist = fig.add_subplot(211)
title = 'LLR with N_RM=' + str(N_RM) + ', N_SW=' + str(N_SW) \
      + ' and random seed ' + str(seed)
dist.set_title(title)
dist.set_xlabel('x')
dist.set_ylabel('P(x) ~ exp[-x^2 / 16^2]')
dist.set_xlim(0, 60)
dist.set_yscale('log')
dist.set_ylim(1e-5, 2)
dist.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0])
dist.grid(True)
dist.errorbar(centers, mean, yerr=err, mfc='none', mec='b', marker='o',
             linestyle='None', ms=3)

# Put relative error below
rel = fig.add_subplot(212)
fig.subplots_adjust(hspace = 0.35)
rel.set_xlabel('x')
rel.set_ylabel('Relative error')
rel.set_xlim(0, 60)
rel.set_yscale('linear')
rel.set_ylim(-1.5, 1.5)
rel.set_yticks(np.arange(-1.5, 1.6, step=0.5))
rel.grid(True)
rel.errorbar(x, dev, yerr=relErr, mfc='none', mec='b', marker='o',
             linestyle='None', ms=3)

# Save it to disk after checking it
#plt.show()
outfile = 'LLR.pdf'
plt.savefig(outfile)
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# Clean up and close down
runtime += time.time()
print("Runtime: %0.1f seconds" % runtime)
# ------------------------------------------------------------------
