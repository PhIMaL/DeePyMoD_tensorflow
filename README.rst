========
DeePyMoD
========

DeepMod is a deep learning based model discovery algorithm which seeks the partial differential equation underlying a spatio-temporal data set. DeepMoD employs sparse regression on a library of basis functions and their corresponding spatial derivatives. 


Description
===========

A feed-forward neural network approximates the data set and automatic differentiation is used to construct this function library and perform regression within the neural network. This construction makes it extremely robust to noise and applicable to small data sets and, contrary to other deep learning methods, does not require a training set and is impervious to overfitting. We illustrate this approach on several physical problems, such as the Burgers', Korteweg-de Vries, advection-diffusion and Keller-Segel equations. 

Note
====

This project has been set up using PyScaffold 3.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
