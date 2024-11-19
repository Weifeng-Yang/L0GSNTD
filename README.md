## Introduction
Nonnegative Tucker decomposition (NTD) is a powerful feature extraction tool widely used in dimensionality reduction and clustering of multi-dimensional data. In this paper,  we propose a novel graph regularized sparse nonnegative Tucker decomposition with $\ell_{0}$-norm constraints ($\ell_{0}$-GSNTD) method. Unlike most existing sparse NTD methods, which overlook the manifold structure of data and uncontrollably promote the sparsity of the core tensor and factor matrices by using a relaxation scheme of $\ell_{0}$-norm regularization, 
our method incorporates the graph regularization into NTD to encode the manifold structure information of data and directly employs the $\ell_{0}$-norm constraints to explicitly control the sparsity of the core tensor and factor matrices in NTD, thereby enhancing the feature extraction capability.  However, due to the nonconvex nature of NTD and the nonconvex and nonsmooth nature of the $\ell_{0}$-norm constraints, optimizing $\ell_{0}$-GSNTD is NP-hard. To tackle these challenges, we propose a Proximal Alternating Linearized (PAL) algorithm to solve the original $\ell_{0}$-GSNTD, and introduce the inertial version of PAL algorithm named inertial PAL (iPAL) algorithm to accelerate convergence. Our algorithms provide a practical convergent scheme to directly solve $\ell_{0}$-GSNTD without relaxing its constraints. Furthermore, we prove that the sequence generated by our algorithms is globally convergent to a critical point and analyze the per iteration complexity of our algorithms. The experimental results on the unsupervised clustering tasks using ten real-world benchmark datasets demonstrate that our method outperforms some state-of-the-art methods.  

This package contains code for the $\ell_0$-GSNTD problem in the paper[<sup>1</sup>](#refer-id). 

## Matlab code
A toy example explains how to use the these function. For "L0GSNTD", before running it, first add the toolbox 'tensortoolbox'[<sup>2</sup>](#refer-id) (www.tensortoolbox.org) to the running path of matlab, and then run the function 'main_Run_me'. 


## Data
This code has built-in some data mentioned in our paper[<sup>1</sup>](#refer-id), and the rest of the data can be downloaded from the mentioned public website. 

## Reference
<div id="refer-id"></div>
[1] Graph regularized sparse nonnegative Tucker decomposition with ℓ0-constraints for unsupervised learning. 

[2] Brett W. Bader and Tamara G. Kolda. 2006. Algorithm 862: MATLAB tensor classes for fast algorithm prototyping. ACM Trans. Math. Softw. 32, 4 (December 2006), 635–653. https://doi.org/10.1145/1186785.1186794
