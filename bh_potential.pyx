#cython: boundscheck=False, wraparound=False, nonecheck=False, language_level=3

from libc.stdlib cimport malloc, free, realloc
from libc.math cimport sqrt
import numpy as np
cimport numpy as np
import matplotlib.pyplot as plt

# constants
cdef int MAX_OCTREE_DEPTH = 16

# offsets used for subdividing octree
cdef double offsets[8][3]
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
    
ctypedef struct NodeData:
    double mass
    double[3] com
    int depth
    double size
    
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
    cdef double G # big G, by default, for units of kpc km^2 / Msun / s^2
    cdef double theta # Barnes-Hut accuracy threshold
    cdef double[:, :] positions
    cdef double[:] masses

    def __init__(self, positions_p: np.ndarray, masses_p: np.ndarray, theta: float = 0.5, G: float = 4.30091e-6):
        """
        Initialize the octree and insert particles into it.
        """
        cdef int i
        cdef int n = positions_p.shape[0]
        cdef double[3] center
        cdef double size
        
        self.n = n
        self.theta = theta
        self.G = G  

        # get centre of mass of all particles
        center[0], center[1], center[2] = np.average(positions_p, axis=0, weights=masses_p)
        
        # bounding box size (twice the actial extent, since the COM isn't exactly 0
        size = np.max(np.ptp(positions_p, axis=0)) * 2.

        # convert input positions and masses to C arrays
        cdef double[:, :] positions = positions_p
        cdef double[:] masses = masses_p
        self.positions = positions
        self.masses = masses

        del positions_p
        del masses_p

        # create the root node of the octree
        self.root = create_octree_node(center, size, 1)

        # insert each particle into the octree
        for i in range(n):
            insert(self.root, &positions[i, 0], masses[i])

        # compute mass and center of mass over the entire tree
        compute_mass_com(self.root)
        
    def tree_info(self, max_search_depth=10):
        """
        Retrieve the properties of each octree node.

        Returns
        -------
        tuple
            A tuple of NumPy arrays containing:
            - node_mass (np.ndarray): Masses of nodes.
            - node_size (np.ndarray): Sizes (side lengths) of nodes.
            - node_cent (np.ndarray): Centers of nodes.
            - node_com (np.ndarray): Centers of mass of nodes.
            - node_depth (np.ndarray): Depth levels of nodes.
            - node_npart (np.ndarray): Number of particles in each node.

        Notes
        -----
        This method collects the mass, size, position, center of mass, depth, and 
        number of particles for all nodes in the octree. It provides a complete 
        snapshot of the tree's structure, which is useful for diagnostics and 
        visualization.
        """
        return collect_tree(self.root, max_search_depth)
    
    def plot_octree_slice(self, max_search_depth=10, z_slice=0):
        """
        Visualize a 2D slice of the octree structure at a specific depth and z-coordinate.

        Parameters
        ----------
        max_search_depth : int, optional
            Maximum depth of the nodes to include in the visualization (default is 8).
        z_slice : float, optional
            z-coordinate of the slice in kpc (default is 0).

        Returns
        -------
        None
            Displays a Matplotlib plot showing the selected slice of the octree.

        Notes
        -----
        This method visualizes the nodes of the octree that overlap with a given 
        z-coordinate slice and are within the specified maximum depth. Each node is 
        represented as a square in the 2D projection, and its size corresponds to 
        the node's spatial extent in the octree.
        """
        # retreive properties of each node
        node_mass, node_size, node_cent, node_com, node_depth, node_npart = collect_tree(self.root, max_search_depth)


        # select nodes above max_search_depth that overlap with the z_slice
        z_min = node_cent[:, 2] - node_size / 2
        z_max = node_cent[:, 2] + node_size / 2
        valid_nodes = (z_min <= z_slice) & (z_max >= z_slice) & (node_depth <= max_search_depth)

        # filter the nodes that meet both conditions
        valid_centers = node_cent[valid_nodes]
        valid_sizes = node_size[valid_nodes]

        fig, ax = plt.subplots(figsize=(8, 8))

        # plot the valid nodes
        for center, size in zip(valid_centers, valid_sizes):
            # create a square for the node
            ax.add_patch(plt.Rectangle((center[0] - size / 2, center[1] - size / 2), 
                                       size, size, linewidth=0.2, 
                                       edgecolor='grey', facecolor='none', alpha=1))

        # set the axis limits using the limits of the particles used to build the tree
        max_pos = (np.max(np.abs(np.asarray(self.positions))))
        ax.set_xlim(-max_pos,max_pos)
        ax.set_ylim(-max_pos,max_pos)

        ax.text(0.98, 0.98, f'Octree Structure (Depth ≤ {max_search_depth})', transform=ax.transAxes,
                fontsize=16, va='top', ha='right')

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
            #compute_potential_parallel(self.root, positions, n_parts, potentials, njobs=njobs)
        else:   
            self.compute_potential_serial(self.root, positions, n_parts, potentials)

        return np.array(potentials)

    def __dealloc__(self):
        """
        free the allocated memory for the octree.
        """
        free_octree(self.root)  # Free the octree structure
        
        
    cdef void compute_potential_serial(self, OctreeNode* root, double[:, :] positions, int n_parts, double[:] potentials):
        '''
        compute the gravitational potential for a batch of particles
        '''
        cdef double[3] r_vec
        cdef double r_sq
        cdef double r
        cdef int i, j

        # allocate a stack array with the maximum possible size of the number of particles used to build the tree
        cdef int stack_capacity = 2 * MAX_OCTREE_DEPTH # heuristic for largest expected stack size
        cdef OctreeNode** stack = <OctreeNode**> malloc(stack_capacity * sizeof(OctreeNode*))
        if stack is NULL:
            raise MemoryError("Failed to allocate initial stack")
        cdef int stack_size = 0
        cdef OctreeNode* current_node

        try:
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
                    if (current_node.size / r) < self.theta:
                        potentials[i] += -self.G * current_node.mass / r 
                    else:
                        for j in range(8):
                            if current_node.children[j] is not NULL:
                                if stack_size >= stack_capacity:
                                    stack_capacity *= 2
                                    stack = <OctreeNode**> realloc(stack, stack_capacity * sizeof(OctreeNode*))
                                    if stack is NULL:
                                        raise MemoryError("Failed to reallocate memory for stack")
                                stack[stack_size] = current_node.children[j]  # push child node to stack
                                stack_size += 1
        finally:
            free(stack)  # free memory allocated for stack
    
cdef OctreeNode* create_octree_node(double[3] center, double size, int depth):
    '''
    creates a new OctreeNode
    '''
    cdef int i
    cdef OctreeNode* node = <OctreeNode*>malloc(sizeof(OctreeNode))
    if node is NULL:
        raise MemoryError("Failed to allocate memory for OctreeNode")
        
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
        raise MemoryError("Failed to allocate memory for child nodes")
    for i in range(8):
        node.children[i] = NULL  # each child initialized as NULL
    
    return node

cdef void insert(OctreeNode* node, double[3] position, double mass):
    '''
    inserts a particle into an OctreeNode
    '''
    # don't go any deeper than than MAX_OCTREE_DEPTH
    if node.depth >= MAX_OCTREE_DEPTH:
        # insert particle into the current node, don't subdivide further
        if node.particles is NULL:
            node.particles = <Particle*>malloc(sizeof(Particle))
            if node.particles is NULL:
                raise MemoryError("Failed to allocate memory for particles in leaf node.")
            node.particles[0].position = position
            node.particles[0].mass = mass
            node.mass = mass
            node.com = position
            node.num_particles = 1
        else:
            # add particle to the existing list
            node.num_particles += 1
            node.particles = <Particle*>realloc(node.particles, node.num_particles * sizeof(Particle))
            if node.particles is NULL:
                raise MemoryError("Failed to reallocate memory for particles in leaf node.")
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
    
cdef tuple collect_tree(OctreeNode* node, int max_search_depth=10, int depth=0):
    '''
    Collect mass, size, position, center of mass, depth, and number of particles for all nodes
    up to a given maximum depth and return them as numpy arrays.
    '''
    # initialize lists to hold the data for each node
    cdef list masses_list = []
    cdef list sizes_list = []
    cdef list positions_list = []
    cdef list coms_list = []
    cdef list depths_list = []
    cdef list num_particles_list = []

    # if the node has no children (leaf node) or the depth exceeds max_search_depth, just return its mass, size, position, COM, depth, and number of particles
    if node.children[0] is NULL or depth > max_search_depth:
        masses_list.append(node.mass)
        sizes_list.append(node.size)
        positions_list.append(tuple(node.center)) 
        coms_list.append(tuple(node.com))
        depths_list.append(depth)
        num_particles_list.append(node.num_particles)
        return (np.array(masses_list), np.array(sizes_list), np.array(positions_list), 
                np.array(coms_list), np.array(depths_list), np.array(num_particles_list))

    cdef int i

    # otherwise, loop over all children and collect mass, size, position, COM, depth, and number of particles
    for i in range(8):
        if node.children[i] is not NULL:
            child_masses, child_sizes, child_positions, child_coms, child_depths, child_num_particles = \
                collect_tree(node.children[i], max_search_depth, depth + 1)
            
            # extend the lists with the new child data
            masses_list.extend(child_masses.tolist())
            sizes_list.extend(child_sizes.tolist())
            positions_list.extend(child_positions.tolist())
            coms_list.extend(child_coms.tolist())
            depths_list.extend(child_depths.tolist())
            num_particles_list.extend(child_num_particles.tolist())

    # append the current node's data to the lists
    masses_list.append(node.mass)
    sizes_list.append(node.size)
    positions_list.append(tuple(node.center))
    coms_list.append(tuple(node.com))
    depths_list.append(depth)
    num_particles_list.append(node.num_particles)

    # convert the lists to numpy arrays before returning
    return (np.array(masses_list), np.array(sizes_list), np.array(positions_list), 
            np.array(coms_list), np.array(depths_list), np.array(num_particles_list))

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
    
def generate_test_distribution(N_pot_main=20000, M_pot_main=1e12, R_pot_main=100, N_small_clusters=12):
    '''
    generates a test distribution consisting of one large central blob and a number of smaller ones
    '''
    positions_main = np.random.normal(0, R_pot_main, (N_pot_main, 3))
    masses_main = np.full(N_pot_main, M_pot_main / N_pot_main)
    cluster_number_main = np.zeros_like(masses_main)
    
    positions_small = []
    masses_small = []
    cluster_number_small = []
    
    for cluster_n in range(N_small_clusters):
        M_pot_small = 10**np.random.uniform(9,10.5)  # Total mass for each small cluster in solar masses
        R_pot_small = np.random.uniform(2,20)    # Standard deviation for the small clusters in kpc
        N_pot_small = int(N_pot_main * (M_pot_small / M_pot_main))  # Number of particles per small cluster

        # generate random center fixed at z=0, as well as positions and masses for each small cluster
        cluster_pos = np.random.uniform(-R_pot_main*5, R_pot_main*5, 2)  # Random x, y in a larger region
        cluster_pos = np.append(cluster_pos, 0)
        cluster_positions = np.random.normal(cluster_pos, R_pot_small, (N_pot_small, 3))
        cluster_masses = np.full(N_pot_small, M_pot_small / N_pot_small)

        positions_small.append(cluster_positions)
        masses_small.append(cluster_masses)
        cluster_number_small.append(np.ones_like(cluster_masses)*cluster_n)

    # combine the positions and masses
    positions = np.vstack([positions_main] + positions_small)
    masses = np.concatenate([masses_main] + masses_small)
    cluster_number = np.concatenate([cluster_number_main] + cluster_number_small)
    
    return masses, positions, cluster_number
