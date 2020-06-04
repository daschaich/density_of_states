#!/usr/bin/python3
import os
import sys
import glob
import time
import numpy as np
from scipy.special import erf, erfinv
import matplotlib.pyplot as plt
# ------------------------------------------------------------------
# Use LLR method to reconstruct the distribution
#   exp[-alpha^2 * x^2] with alpha = 1 / 16

# The exact result for the slope of the log in interval i is just
#   a_i = -2 * alpha^2 * x_i with x_i = i * delta_x
# So with alpha = 1 / 16 we should have a_i = -x_i / 128
# Let's try the initial guess a_i = -0.01 * (x_i + delta / 2)
# and check for convergence to the expected value...

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

# Parse arguments: random seed and N_{RM}
if len(sys.argv) < 4:
  print("Usage:", str(sys.argv[0]), "<RNG seed> <RM steps> <sweeps>")
  sys.exit(1)
seed = int(sys.argv[1])
N_RM = int(sys.argv[2])
N_SW = int(sys.argv[3])
tot_time = -time.time()

# Seed default (PCG64) random number generator
prng = np.random.default_rng(seed)

# Hardcoded input parameters discussed above
alpha = 1 / 16.0
alpha_inv = 16.0
N_B = 1
E_min = 0.0
E_max = 60.0
delta_E = 0.5
delta_ESq = delta_E * delta_E
num_E = 120

# !!!Hack to keep Robbins--Monro recursion from diverging
TOL = 0.5
scale = 9
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# Helper utilities to convert between uniform and transformed random numbers
sqrt_pi_ov_two = 0.5 * np.sqrt(np.pi)

def F(x, a_i):
  ratio = 0.5 * a_i * alpha_inv
  return 0.5 * (1.0 + erf(alpha * x + ratio))

def Finv(u, a_i):
  ratio = 0.5 * a_i * alpha_inv
  return alpha_inv * (erfinv(2.0 * u - 1.0) - ratio)
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# Following Algorithm 1 in section 2.4 of arXiv:1509.08391)
E = np.empty(num_E, dtype = np.float)
dat = np.empty((num_E, N_B), dtype = np.float)
for B in range(N_B):        # Samples for bootstrapping
  runtime = -time.time()
  for i in range(num_E):    # Scanning energy range [0, E_max) by delta_E
    E_lo = i * delta_E
    E[i] = E_lo + 0.5 * delta_E
    E_hi = E_lo + delta_E
    a_i = -0.01 * E[i]      # Initial guess discussed above
#    print("a_i = %.4g at RM iter 0" % a_i)
    for n in range(N_RM):   # Robbins--Monro recursion
      # Generate N_SW values of E distributed as exp[-a_i * E - alpha * E^2]
      # Require E_i <= E < E_p = E_i + delta_E
      # See accompanying tex notes for derivation of F(x) and Finv(u)
      u_lo = F(E_lo, a_i)
      u_hi = F(E_hi, a_i)
      u = prng.uniform(u_lo, u_hi, N_SW)          # N_SW-component vector

      # Check for sensible u_lo & u_hi for debugging
#      if n == 0:
#        print("E_%d in [%.4g, %.4g) --> u in [%.4g, %.4g), width %.4g" \
#              % (i, E_lo, E_hi, u_lo, u_hi, u_hi - u_lo))

      # Compute <<delta_E>>_i(a_i^{(n)})
      Delta_E = 0.0
      nSamples = N_SW
      for j in range(N_SW):
        Delta_E += Finv(u[j], a_i)
      vev_Delta_E = Delta_E / nSamples - E[i]

      # Update a_i and check for convergence to -2alpha * x_i
      # !!!Hack to reduce instabilities for small n...
      #new = a_i + 12.0 * vev_Delta_E / ((n + 1.0) * delta_ESq)
      new = a_i + 12.0 * vev_Delta_E / ((n + 10.0) * delta_ESq)
      diff = np.abs(new - a_i)

      # !!! Hack to (hopefully) cure instabilities with large diff
      # Recompute <<Delta_E>> with more samples (set by 'scale')
      # if diff is too large (set by 'TOL')
      if diff > TOL:
        nSamples += scale * N_SW
        print("Collecting %d samples at RM iter %d " % (nSamples, n), end='')
        print("for E_%d = %.4g, " % (i, E[i]), end='')
        print("to reduce |%.2g - %.2g| = %.2g from <<Delta_E>> = %.2g" \
              % (new, a_i, diff, vev_Delta_E))
        u = prng.uniform(u_lo, u_hi, scale * N_SW)
        for j in range(scale * N_SW):
          E += Finv(u[j], a_i)
        vev_Delta_E = E / nSamples - E[i]
        #new = a_i + 12.0 * vev_Delta_E / ((n + 1.0) * delta_ESq)
        new = a_i + 12.0 * vev_Delta_E / ((n + 10.0) * delta_ESq)
        diff = np.abs(new - a_i)

      # Hopefully things are now stable...
      a_i = new
#      print("a_i = %.4g at RM iter %d, " % (a_i, n + 1), end='')
#      print("with <<Delta_E>> = %.4g" % vev_Delta_E)

    # Recursion complete, so now we have the Bth sample of dat[i]
#    print("a_i for E_i = %.4g converged to %.4g " % (E[i], a_i), end='')
#    print("(vs. %.4g) " % (-2.0 * alpha * alpha * E[i]), end='')
#    print("after %d Robbins--Monro iterations, |delta|=%.2g" % (n + 1, diff))
    dat[i][B] = np.exp(E[i] * a_i)            # TODO: This is wrong...

  # Monitor runtime
  runtime += time.time()
  print("Bootstrap sample %d done in %0.1f seconds" % (B + 1, runtime))
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# Quick visual check of one bootstrap sample, compared with P(x)
# TODO: Plot rho as histogram...
toplot = np.empty_like(E)
for i in range(num_E):
  toplot[i] = dat[i][0]
plt.semilogy(E, toplot, linestyle='None', marker="o")
norm = 2.0 * alpha / np.sqrt(np.pi)   # Double to add both +/- sides
plt.semilogy(E, norm * np.exp(-E**2 * alpha**2), linewidth=2, , color='r')
plt.axis([0.0, 60.0, 1e-7, 0.1])
plt.xticks(np.arange(0, 61, step=20))
title = 'LLR with delta=' + str(delta_E) + ', N_SW=' + str(N_SW) + \
        ' and random seed ' + str(seed)
plt.title(title)
plt.xlabel('x')
plt.ylabel('P(x) ~ exp[-x^2 / 16^2]')
#outfile = 'LLR_check.pdf'
#plt.savefig(outfile)
plt.show()
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# Now go through each the energy range [0, E_max) and bootstrap each delta_E
# Do N_B resamples each selecting N_B points with replacement
# References for bootstrapping:
# www.physics.utah.edu/~detar/phys6720/handouts/ftspect/ftspect/node7.html
# --> The uncertainty is the standard deviation of the distribution
# www.itl.nist.gov/div898/handbook/eda/section3/bootplot.htm
sample = np.empty(N_B, dtype = np.float)  # N_B numbers to be pulled from dat
counts = np.empty((num_E, N_B), dtype = np.float)   # As in naive.py
for i in range(num_E):
  for B in range(N_B):
    # Checked that this covers the full range, with replacement
    indices = prng.integers(0, high=N_B, size=N_B)
    for j in range(N_B):
      sample = dat[i][indices[j]]

    counts[i][B] = sample.mean()

# Now average the counts in each bin
# The error is the standard deviation of the population, not of the mean
mean = counts.mean(0)
err = counts.std(0)
#print(len(mean), len(err))   # 51, 51
#print(mean)
#print(err)
#sys.exit(0)

# Set up relative error --- some shenanigans needed to skip NaNs
xList = []
devList = []                # Relative deviation of results from exact
relList = []
for i in range(num_E):      # Try to work around divide-by-zero warnings
  if mean[i] > 0.0:
    xList.append(centers[i])
    devList.append(mean[i] / np.exp(-centers[i]**2 / stdev**2) - 1.0)
    relList.append(err[i] / mean[i])
    # Check absolute deviations in units of bootstrap errors
#    dev = (mean[i] - np.exp(-centers[i]**2 / stdev**2)) / err[i]
#    print("%.2g %.4g" % (centers[i], dev))
x = np.array(xList)
dev = np.array(devList)
relErr = np.array(relList)
#print(len(x), len(dev), len(relErr))
#print(dev)
#print(relErr)
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
tot_time += time.time()
print("Runtime: %0.1f seconds" % tot_time)
# ------------------------------------------------------------------
