
import numpy as np
import os
import matplotlib.pyplot as plt
from plotting import *


# assign directory
directory = 'accuracies'
 
# iterate over files in
# that directory
plt.figure(figsize=(24, 16))
for filename in os.listdir(directory):
    file = os.path.join(directory, filename)
    # checking if it is a file
    # if os.path.isfile(f):
    #     print(f)

    with open(file, 'rb') as f:

        a = np.load(f, allow_pickle=True)

    #print(f"{file}:  {np.sum(np.sum(a))}")
    accs = a
    avg = np.mean(np.array([accs[0], accs[1], accs[2]]), axis=0)
    plt.title(f"Average accuracy over time (How often correct lable is most frequently predicted class)")
    plt.plot(avg)
    plt.ylim(0,1)
    plt.xlabel("Time (ms), dt=1ms")
    plt.ylabel("Ratio of Correctness")

plt.show()
