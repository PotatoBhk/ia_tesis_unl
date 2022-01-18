# choose a random element from a list
from random import sample, seed
from random import choice
# seed random number generator
seed(1)
# prepare a sequence
sequence = [[1,2],[3,4],[5,6],[7,8]]
print(sequence)

s = sample(sequence, 2)

print(len(s))
print(s)