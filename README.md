# SyntheticSpacePrediction
This is a repository for paper *"Holistic Prediction of Synthetic Space by Implementing Transition State Knowledge in Machine Learning"*. Here, you can find scripts used in this study.
# Introduction
Asymmetric catalysis provides a central strategy for molecular synthesis. Due to the high-dimensional nature of the structure-enantioselectivity relationship, the enantioselectivity prediction in asymmetric catalysis has been a long-standing challenge in synthetic chemistry. The incomprehensive and inaccurate understanding of the synthetic space results in laborious and time-consuming efforts in the optimization of asymmetric reactions, even if the same transformation has already been optimized on model substrates. Here we present a data-driven workflow to achieve holistic prediction of the synthetic space by implementation of transition state knowledge in machine learning. Confirmed in the enantioselectivity prediction of asymmetric palladium-catalyzed electro-oxidative C–H bond activation, the vectorization of transition state knowledge allows for excellent descriptive ability and extrapolation capability of the machine learning model, which enables the quantitative evaluation of the massive synthetic space of 889056 possibilities. This established workflow provides opportunity to harness the hidden value of the widely existing catalysis screening data and transition state model in molecular synthesis, and the created statistical model is able to identify the non-intuitive pattern of structure-performance relationship and make predictions that are challenging for human chemists.
# Dependency
In order to run Jupyter Notebook involved in this repository, several third-party python packages are required. The versions of these packages in our PC are listed below.
```
ase=3.21.0
dscribe=1.0.0
python=3.8.5
numpy=1.19.2
openbabel=3.1.1
pandas=1.3.3
rdkit=2019.09.3
scikit-learn=0.23.2
seaborn=0.11.1
xgboost=1.3.3
```
We suggest using Anaconda to build the python environment, as there are several default packages used in the scripts but not mentioned above. All test are executed under Ubuntu 18.04.
# Usage
Here we provide several notebooks to demonstrate how to vectorize the reaction dataset, perform cross-validation, transform the unbalanced target and predict the whole synthetic space using $\Delta$-learning approach. 

Duo to file size limitations, we don't provide prediction results of the massive synthetic space of 889056 possibilities directly, but rather provide the [notebook](https://github.com/licheng-xu-echo/SyntheticSpacePrediction/blob/main/notebook/3-Holistic-Prediction-over-Synthetic-Space.ipynb) to generate these prections.
# How to cite
The paper is under review.
# Contact with us
Email: hxchem@zju.edu.cn; licheng_xu@zju.edu.cn
