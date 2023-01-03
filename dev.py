import numpy as np
import random as r

blank_list = [0.0]*9
num_rotations = 4
rotation_degrees = np.linspace(0, 360, num_rotations + 1).tolist()[:-1]
thetas = np.random.choice(rotation_degrees, size=4).tolist()
rand_index = r.sample(range(0,9), 4)

for i in range(4):
    blank_list[rand_index[i]] = thetas[i]

print(blank_list)

for i in range(9):
    if blank_list[i] != 0.0:
        print(blank_list[i])