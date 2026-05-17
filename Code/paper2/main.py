# log(sys.argv)
# From sys arguments
# 1 filepath
# 2 simulated 1 or 0
# 3 no of OTUs
# 4 population size
# 5 number of generations
# 6 print each n generation : n


import sys
from dataset import Dataset

dataset = Dataset.from_csv(str(sys.argv[1]))
