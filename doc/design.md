# DESIGN DOC

This document includes design choice and detailed document of all fucntions and classes in my project.

## 1. General Design

### 1.1 parameter settlement

`main.py` includes the `__main__` function of whole project.
It will set experiment parameters(e.g. number of rounds, learning rate) by argparse module and encapsulate all params in a class, and return it in a tuple by calling a function in this class.

###  1.2 dataloader

For `MNIST` dataset, we use  `torchvision.datasets` to load dataset, 


### 1.3 training

Then training dataset will be passed to `federated_learning` along with all params mentioned above(T, num of clients, E, B).

This part will return final accuracy and print it on console.


# 2. Training
`federated_learning` will create `ServerOPT` object when called by passing params mentioned above.
This part will use `torch.utils.data` to make dataset partition for all clients.

