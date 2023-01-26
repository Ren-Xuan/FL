# Anonymous repository

------

#### We have removed any information in the code that might contain the identity of the authors

------

#### Run Code:

python main_fed.py --iid   --gpu 0 --local_ep 5 --local_bs 10  --num_users 1024 --frac 0.1 --dataset [fashionmnist,cifar,agnews]  --model [cnn,resnet,lstm] --epochs 250 [--fedslice]

num_users represents the number of edge devices

- **'num_users'**  represents the total number of edge devices 
- **'frac' ** represents  the proportion of edge devices participating in the training each epoch
- This parameter **'fedslice'** means that the FedSlice algorithm is used; without this parameter, the FedAvg algorithm is used.

------

#### Note: 

#### 		Our code implementation only simulates the federated learning environment.

#### 	Code implementation of the main Federated Learning structure modified from: 

Shaoxiong Ji. (2018, March 30). A PyTorch Implementation of Federated Learning. Zenodo. http://doi.org/10.5281/zenodo.4321561

​			
