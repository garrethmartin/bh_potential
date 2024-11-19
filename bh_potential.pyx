#cython: boundscheck=False, wraparound=False, nonecheck=False, language_level=3

from libc.stdlib cimport malloc, free, realloc
from libc.math cimport sqrt
import numpy as np
cimport numpy as np


cdef double G = 4.30091e-6 # big G for units of kpc km^2 / Msun / s^2
cdef double THETA = 0.5  # Barnes-Hut accuracy threshold
cdef int MAX_OCTREE_DEPTH = 16

cdef double offsets[8][3] # offsets for building octree
offsets[:] = [
    [1,  1,  1],
    [1,  1, -1],
    [1, -1,  1],
    [1, -1, -1],
    [-1,  1,  1],
    [-1,  1, -1],
    [-1, -1,  1],
    [-1, -1, -1],
]

cdef int max_depth = 1 # store the maximum depth of the tree

# C struct for Particle
cdef struct Particle:
    double[3] position
    double mass

# C struct for OctreeNode
ctypedef struct OctreeNode:
    double[3] center  # node center
    double size       # node side length
    double mass       # total mass in node
    double[3] com     # node COM
    int num_particles # number of particles in node
    int depth         # node depth
    Particle* particles  # list of particles in the node
    OctreeNode** children  # list of node children

cdef OctreeNode* create_octree_node(double[3] center, double size, int depth):
    '''
    creates a new OctreeNode
    '''
    cdef int i
    cdef OctreeNode* node = <OctreeNode*>malloc(sizeof(OctreeNode))
        
    node.center[0] = center[0]
    node.center[1] = center[1]
    node.center[2] = center[2]
    node.size = size
    node.mass = 0.0
    node.com[0] = 0.0
    node.com[1] = 0.0
    node.com[2] = 0.0
    node.num_particles = 0
    node.depth = depth
    node.particles = NULL
    node.children = <OctreeNode**>malloc(8 * sizeof(OctreeNode*))  # allocate memory for 8 child nodes
    if node.children is NULL:
        free(node)
    
    for i in range(8):
        node.children[i] = NULL  # each child initialized as NULL
    
    return node

cdef void free_octree(OctreeNode* node):
    '''
    frees memory recursively
    '''
    cdef int i
    
    if node is NULL:
        return
    for i in range(8):
        if node.children[i] is not NULL:
            free_octree(node.children[i])
    if node.particles is not NULL:
        free(node.particles)
    free(node.children)
    free(node)

cdef void insert(OctreeNode* node, double[3] position, double mass):
    '''
    inserts a particle into an OctreeNode
    '''
    # don't go any deeper than than MAX_OCTREE_DEPTH
    if node.depth >= MAX_OCTREE_DEPTH:
        # insert particle into the current node, don't subdivide further
        if node.particles is NULL:
            node.particles = <Particle*>malloc(sizeof(Particle))
            node.particles[0].position = position
            node.particles[0].mass = mass
            node.mass = mass
            node.com = position
            node.num_particles = 1
        else:
            # add particle to the existing list
            node.num_particles += 1
            node.particles = <Particle*>realloc(node.particles, node.num_particles * sizeof(Particle))
            node.particles[node.num_particles - 1].position = position
            node.particles[node.num_particles - 1].mass = mass
            node.mass += mass

        return
    
    # if depth is not too large, allow subdivision
    if node.children[0] is NULL:
        _subdivide(node)
        _reinsert_existing_particles(node)
        _insert_in_child(node, position, mass)
    else:
        _insert_in_child(node, position, mass)

cdef void _subdivide(OctreeNode* node):
    '''
    subdivides an OctreeNode
    '''
    cdef double half_size = node.size / 2
    cdef double[3] new_center
    cdef int i, j
        
    # create 8 children nodes
    for i in range(8):
        for j in range(3):
            new_center[j] = node.center[j] + offsets[i][j] * (half_size / 2)
        node.children[i] = create_octree_node(new_center, half_size, node.depth + 1)

cdef void _reinsert_existing_particles(OctreeNode* node):
    '''
    reinsert existing particles into the new child nodes
    '''
    cdef Particle particle
    cdef int i
    
    for i in range(node.num_particles):
        particle = node.particles[i]
        _insert_in_child(node, particle.position, particle.mass)
    # free memory allocated for particles
    free(node.particles)
    node.particles = NULL
    node.num_particles = 0

cdef void _insert_in_child(OctreeNode* node, double[3] position, double mass):
    '''
    insert a particle into a child node
    '''
    cdef double diff
    cdef OctreeNode* child
    cdef int i, j
    
    for i in range(8):
        child = node.children[i]
        if child is NULL:
            continue
        # check if the particle is within the bounds of the child cell
        for j in range(3):
            diff = position[j] - child.center[j]
            if abs(diff) > child.size / 2:
                break
        else:
            insert(child, position, mass)
            child.num_particles += 1
            break


cdef void compute_mass_com(OctreeNode* node):
    '''
    update mass and COM of all nodes
    '''
    if node.children[0] is NULL:
        return  # no need to do anything for leaf node

    cdef double total_mass = 0.0
    cdef double[3] weighted_com = [0.0, 0.0, 0.0]
    cdef int i

    # loop over all children and accumulate mass and COM
    for i in range(8):
        if node.children[i] is not NULL:
            compute_mass_com(node.children[i])

            total_mass += node.children[i].mass
            for j in range(3):
                weighted_com[j] += node.children[i].mass * node.children[i].com[j]

    if total_mass > 0:
        for i in range(3):
            node.com[i] = weighted_com[i] / total_mass

    node.mass = total_mass

cdef void compute_potential_serial(OctreeNode* root, double[:, :] positions, int n_parts, double[:] potentials):
    '''
    compute the gravitational potential for a batch of particles
    '''
    cdef double[3] r_vec
    cdef double r_sq
    cdef double r
    cdef int i, j

    # allocate a stack array with the maximum possible size of the number of particles used to build the tree
    cdef OctreeNode** stack = <OctreeNode**> malloc(MAX_OCTREE_DEPTH * 10 * sizeof(OctreeNode*))
    cdef int stack_size = 0
    cdef OctreeNode* current_node

    for i in range(n_parts):
        stack_size = 0  # reset stack size

        # push the root node onto stack
        stack[stack_size] = root
        stack_size += 1

        while stack_size > 0:
            current_node = stack[stack_size - 1]  # pop the current node
            stack_size -= 1

            if current_node.mass == 0.0:
                continue

            # distance between particle and node COM
            for j in range(3):
                r_vec[j] = positions[i, j] - current_node.com[j]

            r_sq = r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1] + r_vec[2] * r_vec[2]
            
            if r_sq == 0:  # skip self-interaction
                continue

            r = sqrt(r_sq)
            
            # apply the Barnes-Hut approximation (if needed)
            if (current_node.size / r) < THETA:
                potentials[i] += -G * current_node.mass / r 
            else:
                for j in range(8):
                    if current_node.children[j] is not NULL:
                        stack[stack_size] = current_node.children[j]  # push child node to stack
                        stack_size += 1

    free(stack)  # free memory allocated for stack
    
                
cdef class Octree:
    """
    Initialize the octree and insert particles into it.

    Parameters
    ----------
    positions_p : np.ndarray
        A 2D NumPy array of shape (n_particles, 3) containing the positions 
        of particles in 3D space in units of kpc.
    masses_p : np.ndarray
        A 1D NumPy array of shape (n_particles,) containing the masses 
        of the particles in units of solar masses.

    Notes
    -----
    This function creates the root node of the octree, inserts all particles, 
    and calculates the mass and center of mass for the entire tree.
    """
    cdef OctreeNode* root
    cdef int n  # number of particles in tree

    def __init__(self, positions_p: np.ndarray, masses_p: np.ndarray):
        """
        Initialize the octree and insert particles into it.
        """
        cdef int i
        cdef int n = positions_p.shape[0]
        cdef double[3] center
        cdef double size
        
        self.n = n

        # get centre of mass of all particles
        center[0], center[1], center[2] = np.average(positions_p, axis=0, weights=masses_p)
        
        # bounding box size (twice the actial extent, since the COM isn't exactly 0
        size = np.max(np.ptp(positions_p, axis=0)) * 2.

        # convert input positions and masses to C arrays
        cdef double[:, :] positions = positions_p
        cdef double[:] masses = masses_p

        del positions_p
        del masses_p

        # create the root node of the octree
        self.root = create_octree_node(center, size, 1)

        # insert each particle into the octree
        for i in range(n):
            insert(self.root, &positions[i, 0], masses[i])

        # compute mass and center of mass over the entire tree
        compute_mass_com(self.root)

    def compute_potentials(self, positions_p: np.ndarray, njobs: int = 1):
        """
        Calculate the gravitational potential at specified positions.

        Parameters
        ----------
        positions_p : np.ndarray
            A 2D NumPy array of shape (n_points, 3) containing the positions 
            where the gravitational potential should be calculated in units of kpc.
        njobs : int, optional
            Number of parallel jobs to use for the calculation (default is 1).
            Currently, parallel computation (`njobs > 1`) is not implemented.

        Returns
        -------
        np.ndarray
            A 1D NumPy array of shape (n_points,) containing the gravitational 
            potential at each input position.

        Notes
        -----
        The gravitational potential is calculated using the Barnes-Hut approximation.
        For now, only single-threaded execution is supported.
        """
        cdef int n = self.n  # Use same n from the initial octree build
        cdef int n_parts = len(positions_p)
        
        # convert input positions to C array
        cdef double[:, :] positions = positions_p
        del positions_p
        cdef double[:] potentials = np.zeros(n_parts, dtype=np.float64) # array to hold potentials
        
        if njobs > 1:
            raise NotImplementedError("njobs > 1 not implemented yet.")
            return
            #compute_potential_parallel(self.root, positions, n_parts, potentials, njobs=njobs)
        else:   
            compute_potential_serial(self.root, positions, n_parts, potentials)

        return np.array(potentials)
    
    def __dealloc__(self):
        """
        free the allocated memory for the octree.
        """
        free_octree(self.root)  # Free the octree structure