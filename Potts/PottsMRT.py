#!/usr/bin/python
import os
import sys
import glob
import time
import numpy as np
from utils import follow_bond
# ------------------------------------------------------------------
# Density of states for 3d three-state Potts model
# LLR method with Robbins--Monro recursion needs importance sampling updates
# Here just using Metropolis--Rosenbluth--Teller algorithm
# With weight exp[-S] = exp[gamma sum_<ij> \delta_{s_i, s_j}]
#   the acceptance probability is exp[gamma * (new# - cur#)]

# Parse arguments: 3d lattice volume,
# Potts coupling gamma, number of sweeps to do, RNG seed
# and directory for output data
if len(sys.argv) < 8:
  print "Usage:", str(sys.argv[0]), "<nx> <ny> <nz>"
  print "                     <gamma> <sweeps> <RNG seed> <out_dir>"
  sys.exit(1)
nx = np.uint(sys.argv[1])
ny = np.uint(sys.argv[2])
nz = np.uint(sys.argv[3])
vol = nx * ny * nz
Ndim = 3                      # Number of dimensions
Ndir = 2 * Ndim               # Number of directions (forward and backward)
Nstate = 3                    # Hard-code three-state Potts model
gamma = float(sys.argv[4])
Nsweep = int(sys.argv[5])
seed = int(sys.argv[6])
outdir = sys.argv[7]
runtime = -time.time()

# TODO: Utilities for loading configuration...

# Create output directory if it doesn't exist already
if not os.path.isdir(outdir):
  print "Creating directory", outdir, "for output"
  os.makedirs(outdir)

# Save run parameters for posterity
PARAMS = open(outdir + '/params.txt', 'w')
print >> PARAMS, "python", ' '.join(sys.argv)

# Seed (Mersenne Twister) random number generator
# Use RandomState instead of (global) seed
# in case multiple independent streams may be needed in the future
prng = np.random.RandomState(seed)
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# Set up lattice
# First define arrays to store (x, y, z) indices of each site
x = np.empty(vol, dtype=np.uint)
y = np.empty(vol, dtype=np.uint)
z = np.empty(vol, dtype=np.uint)
for i, j, k in np.ndindex((nx, ny, nz)):
  index = np.uint(i + nx * (j + ny * k))
  x[index] = i
  y[index] = j
  z[index] = k

# Pack constant information into single variable for passing to subroutines
lattice = dict({'nx': nx, 'ny': ny, 'nz': nz, 'Ndim': Ndim, 'Ndir': Ndir,
                'vol': vol, 'prng': prng, 'x': x, 'y': y, 'z': z})

# Now for each site we need the following:
#   The state of the Potts 'spin'
# We start with randomly assigned values
config = np.empty(vol, dtype=np.uint)
for i in range(vol):
  config[i] = prng.randint(0, Nstate)
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# Open files for output
ACCEPT = open(outdir + '/accept.csv', 'w')
print >> ACCEPT, "sweep,accept"
MAGNET = open(outdir + '/magnet.csv', 'w')
print >> MAGNET, "sweep,state1,state2,state3"
ACTION = open(outdir + '/action.csv', 'w')
print >> ACTION, "sweep,action_tot,action_rel"

# Print starting state
# Note S = -gamma sum_<ij> delta_{s_i, s_j}    # TODO: Check sign...
magnet = [0, 0, 0]
tot_act = 0.0
for i in range(vol):
  magnet[config[i]] += 1    # Count how many sites have each value
  for mu in range(Ndim):    # Only the forward neighbors
    if config[i] == config[follow_bond(i, mu, lattice)]:
      tot_act -= gamma

# Print 'magnetization' and action,
# for each including both total and average over lattice volume
m1 = float(magnet[0]) / float(vol)
m2 = float(magnet[1]) / float(vol)
m3 = float(magnet[2]) / float(vol)
print >> MAGNET, "0,%.8g,%.8g,%.8g" % (m1, m2, m3)
print >> ACTION, "0,%.8g,%.8g" % (tot_act, tot_act / float(vol))

# Loop over sweeps, printing some basic data after each one
for sweep in range(1, Nsweep + 1):
  # Each sweep loops (randomly) over the lattice volume
  accept = 0.0                    # Initialize acceptance rate
  for i in range(vol):
    # Update: Try to change the state at the current site
    # The new state is allowed to be the current state
    ran = prng.randint(0, vol)
    cur = config[ran]
    new = prng.randint(0, Nstate)   # Proposed new state at site ran

    # Compute change in energy, if non-zero
    # With weight exp[-S] = exp[gamma sum_<ij> delta_{s_i, s_j}]
    #   accept with probability exp[diff] = exp[oldE - newE]
    if new == cur:
      accept += 1.0
    else:         # We know new != cur
      diff = 0.0
      for mu in range(Ndir):
        neigh = config[follow_bond(ran, mu, lattice)]
        if new == neigh:
          diff += gamma
        elif cur == neigh:
          diff -= gamma

      if diff > 0:
        config[ran] = new
        accept += 1.0
      elif prng.uniform(0, 1) < np.exp(diff):
        config[ran] = new
        accept += 1.0

  # Print some basic data after each sweep
  # (Can also run after each update if speed is not an issue)
  magnet = [0, 0, 0]
  tot_act = 0.0
  for i in range(vol):
    magnet[config[i]] += 1    # Count how many sites have each value
    for mu in range(Ndim):    # Only the forward neighbors
      if config[i] == config[follow_bond(i, mu, lattice)]:
        tot_act -= gamma

  # Print acceptance, 'magnetization' and action,
  # for each including both total and average over lattice volume
  print >> ACCEPT, "%d,%.4g" % (sweep, accept / float(vol))
  m1 = float(magnet[0]) / float(vol)
  m2 = float(magnet[1]) / float(vol)
  m3 = float(magnet[2]) / float(vol)
  print >> MAGNET, "%d,%.8g,%.8g,%.8g" % (sweep, m1, m2, m3)
  print >> ACTION, "%d,%.8g,%.8g" % (sweep, tot_act, tot_act / float(vol))
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# Clean up and close down
ACCEPT.close()
MAGNET.close()
ACTION.close()

# TODO: Utilities for saving configuration...

runtime += time.time()
print "Runtime: %0.1f seconds" % runtime
print >> PARAMS, "Runtime: %0.1f seconds" % runtime
PARAMS.close()
# ------------------------------------------------------------------
