# cNN-DP: Composite neural network with differential propagation
![github](https://github.com/hyeonbeenlee/cNN-DP/assets/78078652/58c8f2a1-196f-4fa7-8702-adb31e6aae5d)

Hyeonbeen Lee, Seongji Han, Hee-Sun Choi, Jin-Gyun Kim, cNN-DP: Composite neural network with differential propagation for impulsive nonlinear dynamics, Journal of Computational Physics, Volume 496, 2024, 112578, ISSN 0021-9991, https://doi.org/10.1016/j.jcp.2023.112578. (https://www.sciencedirect.com/science/article/pii/S0021999123006733)
## What is this for?
> cNN-DP: Composite neural network with differential propagation

We propose a novel composite neural network for effective learning of highly impulsive/oscillatory/chaotic dynamic systems. We assume our target as **high-order derivatives** and utilize **additional low-order derivatives**, which were *wasted* or *inactively considered* before.  
It is effective, specifically for:
* Modeling oscillatory solutions (derivatives) of differential equations
* Learning noisy and impulsive time series measurements

Examples we present in our paper cover following systems:

* Solutions of chaotic dynamic systems
* Earthquake measurements
* Rigid body contact
* Industrial-level vehicle simulation


## What's the advantage of using this?
* Highly precise generalization on oscillatory, impulsive, and chaotic systems
* Fast training and inference time
* Low VRAM usage
* Robust to data quality and hyperparameters
* Does not require any physical equations to use
* Expandable to any domains, such as functions of space, frequency, or any arbitrary variables


## How does it work?
<img width="1108" alt="Screenshot 2023-06-11 at 20 33 29" src="https://github.com/KHU-MASLAB/cNN-DP/assets/78078652/e640be65-35b1-4f9a-8095-7b755f0eaaf7">

Dynamics tend to become more 'impulsive' in high-order derivatives. In reverse, it becomes 'simpler'.  
Then, why don't we let the neural network also refer to 'simple' when it's learning 'impulsive', rather than solely learning the 'impulsive'? 

We intend our neural network to learn the 'simple' and the 'impulsive' simultaneously by interconnecting multiple MLP subnetworks with corresponding multi-order derivative losses. The preceding subnets predict the 'simple's, and their outputs are connected to inputs of subsequent subnets. This enables richer context information in the learning of impulsive or chaotic systems as functions of time and design variables.

Using ```torch``` pseudocode, the process can be expressed as:
```
y=n_dp0(x) # MLP forward 1
yDot=n_dp1(x,y) # MLP forward 2
yDDot=n_dp2(x,y,yDDot) # MLP forward 3
```
This simple composition of subnetworks along with augmented multi-order derivatives produce strong improvements while maintaining speeds and costs.  
Although we only use three subnets in our paper, the number of subnets in the cNN-DP is **not strictly limited**. Theoretically 2 to ```inf```.

## What is the auto-gradient network?
It is another competitor of the proposed cNN-DP using automatic differentiation to learn multi-order derivatives.  
Suppose we have data of multiple orders of derivatives and we target the highest derivative, just like the cNN-DP. The idea of the auto-gradient network is to utilize automatic differentiation (```torch.autograd.grad```) to compute high-order predictions of neural networks.

It first outputs the lowest-order prediction. Then, we repeatedly differentiate the network to the time variable (which should be given in the input) to reach the highest-order prediction.

This approach can be a considerable substitute. However, it turns out to be extremely expensive and slow for both training and inference.

## How do I use the codes?
In the ```examples``` directory, we have **three examples presented in the paper** each including **data generation, training, and visualizing** codes.

Simply running the ```datagen.py``` will automatically construct data files in the ```data``` directory.

```train.py``` will train network models and save them to ```models``` directory. Inference of saved models can be easily obtained through ```architectures.interface.NetInterface``` class's ```predict``` method as following pseudocode:
```
n_dp = NetInterface(models/SAVED_MODEL.pt)
y,yDot,yDDot=n_dp.predict(input)
```

## Citing the paper (BibTeX)
```
@article{lee2024cnn,
  title={cNN-DP: Composite neural network with differential propagation for impulsive nonlinear dynamics},
  author={Lee, Hyeonbeen and Han, Seongji and Choi, Hee-Sun and Kim, Jin-Gyun},
  journal={Journal of Computational Physics},
  volume={496},
  pages={112578},
  year={2024},
  publisher={Elsevier}
}
```

