#!/usr/bin/python
import os
import sys
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
# ------------------------------------------------------------------
# Generate 100k gaussian random numbers, 'dat', distributed according to
#   exp[-x^2 / 16^2]

# Consider absolute value that we'll also focus on with the LLR method

# Bootstrap dat to determine the relative error on the histogram counts
# in each of Nbins bins (normalized by the first count at x=0)

# !!! Currently only have matplotlib v1.5.2 for python3
# !!! So stick with python2 (and matplotlib v2.2.3) for now

# Parse argument: random seed
if len(sys.argv) < 2:
  print("Usage:", str(sys.argv[0]), "<RNG seed>")
  sys.exit(1)
seed = int(sys.argv[1])
runtime = -time.time()

# Seed (Mersenne Twister) random number generator
# Use thread-safe RandomState instead of (global) seed
# Former should allow multiple independent streams if desired
prng = np.random.RandomState(seed)

# Number of bins for histograms
Nbins = 51
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# Set up array of numbers with hard-coded length Npts=100,000
# We want P(x) ~ exp[x^2 / 16^2]
# Numpy sets up  exp[x^2 / 2sigma^2]
mean = 0.0
stdev = 16.0
Npts = int(1e5)
sigma = stdev / np.sqrt(2.0)
dat = np.abs(prng.normal(mean, sigma, Npts))

# Manually set up bin list using dat
maxdat = max([np.ceil(max(dat)), -1.0 * np.floor(min(dat))])
#print max(dat), np.ceil(max(dat)), min(dat), np.floor(min(dat)), maxdat
spacing = maxdat / Nbins
binlist = np.arange(0, maxdat + 1, spacing)
centers = np.arange(0.5 * spacing, maxdat + 1 - 0.5 * spacing, spacing)
#print len(binlist), len(centers)    # 52, 51
#print binlist
#print centers
#sys.exit(0)

# Quick visual check of full data set
# Use 'density' to compare with P(x)
# This seems to set the integral to unity
# --> will have bins with less than 1/Npts = 1e-5...
#count, bins, ignored = plt.hist(dat, Nbins, density=True,
#                                histtype='step', hatch='//')
#norm = 2.0 / (stdev * np.sqrt(np.pi))   # Double to add both +/- sides
#plt.semilogy(bins, norm * np.exp(-bins**2 / stdev**2),
#             linewidth=2, color='r')
#plt.axis([0.0, 60.0, 1e-7, 0.1])
#plt.xticks(np.arange(0, 61, step=20))
#title = str(Npts) + ' samples with random seed ' + str(seed)
#plt.title(title)
#plt.xlabel('x')
#plt.ylabel('P(x) ~ exp[-x^2 / 16^2]')
#outfile = 'naive_check.pdf'
#plt.savefig(outfile)
#plt.show()

# References for bootstrapping:
# www.physics.utah.edu/~detar/phys6720/handouts/ftspect/ftspect/node7.html
# --> The uncertainty is the standard deviation of the distribution
# www.itl.nist.gov/div898/handbook/eda/section3/bootplot.htm
# --> Do 500 resamples each selecting Npts points with replacement
Nsamples = 500
sample = np.empty_like(dat)   # Npts numbers to be pulled from dat
counts = np.empty((Nsamples, Nbins), dtype = np.float)
for i in range(Nsamples):
  # Checked that this covers the full range, with replacement
  indices = prng.randint(0, high=Npts, size=Npts)
  for j in range(Npts):
    sample[j] = dat[indices[j]]

  # Bin this sample using the fixed binlist defined above
  # Normalize here by count in first bin at x=0
  count, ignored = np.histogram(sample, bins=binlist, density=False)
  norm = 1.0 / float(count[0])
  counts[i] = norm * count
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# Now average the counts in each bin
# The error is the standard deviation of the population, not of the mean
mean = counts.mean(0)
err = counts.std(0)
#print len(mean), len(err)    # 51, 51
#print mean
#print err
#sys.exit(0)

# Set up relative error --- some shenanigans needed to skip NaNs
# Skip first bin, which has no fluctuations since used to normalize
xList = []
devList = []                # Relative deviation of results from exact
relList = []
for i in range(1, Nbins):   # Try to work around divide-by-zero warnings
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
title = str(Npts) + ' samples with random seed ' + str(seed)
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
outfile = 'naive.pdf'
plt.savefig(outfile)
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# Clean up and close down
runtime += time.time()
print("Runtime: %0.1f seconds" % runtime)
# ------------------------------------------------------------------
