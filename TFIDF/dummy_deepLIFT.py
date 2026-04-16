import torch 
from torch import nn 
from torch.nn import functional as F 
import numpy as np 

from logging import getLogger
from captum.attr import IntegratedGradients 


# init the logger 
logger = getLogger(__name__)


# testing if cuda is available 
if torch.cuda.is_available():
    logger.info("CUDA is available.Using GPU.")
else:
    logger.info("CUDA is not available.Using CPU.")



class MyNetWork(nn.Module): 

    def __init__(self): 
        super(MyNetWork, self).__init__() 
        self.linear1 = nn.Linear(10, 20) 
        self.linear2 = nn.Linear(20, 1) 

    def forward(self, x): 
        x = F.relu(self.linear1(x)) 
        x = self.linear2(x) 
        return x
    



dummyModel = MyNetWork()
dummyModel.eval()


# Make computations deterministic, by fixing random seeds.
torch.manual_seed(123)
np.random.seed(123)

# Define input and baseline tensors. 
input = torch.rand(1, 10, 20)
baseline = torch.zeros(1, 10, 20)



# Select algorithm to instanciate and apply (Integrated Gradients in this case).
ig = IntegratedGradients(dummyModel)
attributions, delta = ig.attribute(input, baseline, target=0, return_convergence_delta=True)
print('IG Attributions:', attributions)
print('Convergence Delta:', delta)



exit(0)

