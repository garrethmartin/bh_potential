# **README** for `bh_potential`

## Purpose:

`bh_potential` is a Python extension that computes gravitational potentials for a group of particles using the *Barnes & Hut (1686)* octree method. This approach significantly reduces computational costs compared to direct N-body calculations by approximating distant interactions.

## **Installation**

To build the extension, run the following command in the root directory:

```
python setup.py build_ext --inplace
```
## Usage:

### Initialize the octree:

The Octree class takes particle positions and masses as input, constructs the octree, and computes the mass and center of mass for all nodes:
```
import bh_potential as bh

octree = bh.Octree(positions, masses)
```

`positions`: A NumPy array of shape (n_particles, 3) representing particle positions in 3D space in units of kpc.
`masses`: A NumPy array of shape (n_particles,) containing the masses of the particles in units of solar masses.

### Compute gravitational potentials at new positions
Use the compute_potentials method to calculate the gravitational potential at specified positions:
```
potentials = octree.compute_potentials(test_positions)
```

`test_positions`: A NumPy array of shape (n_points, 3) containing the positions where the potential should be evaluated.

Returns: A NumPy array of shape (n_points,) with the computed gravitational potentials.

## Example:

```
import numpy as np
import bh_potential as bh

# Create normally distributed test data
N_pot = 20000 # number of particles
M_pot = 3e12 # total mass in m_sub
R_pot = 100 # standard deviation in kpc
positions = np.random.normal(0, R_pot, (N_pot, 3))
masses = np.asarray([M_pot/N_pot]*N_pot)

# Build the octree
octree = bh.Octree(positions, masses)

# Evaluate the potential at the same positions
potentials = octree.compute_potentials(positions)
```