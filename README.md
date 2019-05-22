# DeePyMoD

DeepMod is a deep learning based model discovery algorithm which seeks the partial differential equation underlying a spatio-temporal data set. DeepMoD employs sparse regression on a library of basis functions and their corresponding spatial derivatives. This code is based on the paper: [arXiv:1904.09406](http://arxiv.org/abs/1904.09406) 

A feed-forward neural network approximates the data set and automatic differentiation is used to construct this function library and perform regression within the neural network. This construction makes it extremely robust to noise and applicable to small data sets and, contrary to other deep learning methods, does not require a training set and is impervious to overfitting. We illustrate this approach on several physical problems, such as the Burgers', Korteweg-de Vries, advection-diffusion and Keller-Segel equations. 

# Examples
In the notebook folder we present three examples:
* Burgers' equation
* Korteweg-de Vries equation
* 2D Advection-Diffusion equation

**The Burger's example contains detailed instructions on using DeepMoD.** More examples will be uploaded soon ... 

# How to install
We suggest placing DeePyMoD in its own conda environment. Then, to install DeePyMoD simply clone the repository either by downloading the zip or cloning it through git, 

`git clone https://github.com/PhIMaL/DeePyMoD`

and then go to the directory and run:

`python setup.py install`

We haven't included a requirements.txt, so you'll have to satisfy those yourself (Numpy, matplotlib and Tensorflow 1.12/1.13 should work).


