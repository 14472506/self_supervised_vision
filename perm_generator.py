import itertools
import numpy as np
import random
from tqdm import tqdm
from scipy.spatial.distance import hamming

def generate_permutation_set(rot_list, num_permutations, method="maximal"):
    """
    Details
    """ 

    if method not in ["maximal", "average", "minimal"]:
        raise ValueError("The specific method=%s is not recoginised!" % method)

    permutations = []

    # get all permutations
    all_permutations = list(itertools.permutations(rot_list))

    # convert all permutations to 2D matrix
    all_permutations = np.array(all_permutations).T

    # uniformly sample out of (len(rot_list)) indeces to initialise
    current_index = random.randint(0, np.math.factorial(len(rot_list)) - 1)

    for i in tqdm(range(1, num_permutations + 1), desc='Generating Permutation Set'):
        # adding permutations at current index to set and removing
        # current permutations at current index from all permutations
        permutations.append(tuple(all_permutations[:, current_index]))
        all_permutations = np.delete(all_permutations, current_index, axis=1)

        # uniformly sample if average and skip computation
        if method == "average":
            current_index = random.randint(0, np.math.factorial(len(rot_list)) - i)
            continue
       
        # compute the hamming distance matrix
        distances = np.empty((i, np.math.factorial(len(rot_list)) - i))
    
        for j in range(i):
            for k in range(np.math.factorial(len(rot_list)) - i):
                distances[j, k] = hamming(permutations[j], all_permutations[:, k])
        
        distances = np.matmul(np.ones((1, i)), distances)
    
        # choose the next permutation s.t. it maximises objective
        if method == "maximal":
            current_index = np.argmax(distances) 
        elif method == "minimal":
            current_index = np.argmin(distances)
    
    # compute minimum hamming distance in generated permutation sets
    distances_ = []
    for i in range(num_permutations):
        for j in range(num_permutations):
            if i != j:
                distances_.append(hamming(np.array(permutations[i]), np.array(permutations[j])))
    
    min_distance = min(distances_)
    print('Minimum hamming distance is chosen as %0.4f' % min_distance)
    
    print(permutations)
    #return permutations

if __name__ == "__main__":
    rot_list = [0, 1, 2, 3, 0, 0, 0, 0, 0]
    generate_permutation_set(rot_list, 100)