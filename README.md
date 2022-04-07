# SyntheticSpacePrediction
This is a repository for paper *"Holistic Prediction of Synthetic Space by Implementing Transition State Knowledge in Machine Learning"*. Here, you can find scripts used in this study.
# Introduction
Asymmetric catalysis provides a central strategy for molecular synthesis. Due to the high-dimensional nature of the structure-enantioselectivity relationship, the enantioselectivity prediction in asymmetric catalysis has been a long-standing challenge in synthetic chemistry. The incomprehensive and inaccurate understanding of the synthetic space results in laborious and time-consuming efforts in the optimization of asymmetric reactions, even if the same transformation has already been optimized on model substrates. Here we present a data-driven workflow to achieve holistic prediction of the synthetic space by implementation of transition state knowledge in machine learning. Confirmed in the enantioselectivity prediction of asymmetric palladium-catalyzed electro-oxidative C–H bond activation, the vectorization of transition state knowledge allows for excellent descriptive ability and extrapolation capability of the machine learning model, which enables the quantitative evaluation of the massive synthetic space of 889056 possibilities. This established workflow provides opportunity to harness the hidden value of the widely existing catalysis screening data and transition state model in molecular synthesis, and the created statistical model is able to identify the non-intuitive pattern of structure-performance relationship and make predictions that are challenging for human chemists.
# System requirements
In order to run Jupyter Notebook involved in this repository, several third-party python packages are required. The versions of these packages in our station are listed below.
```
ase=3.21.0
dscribe=1.0.0
python=3.7.13
numpy=1.19.2
openbabel=3.1.1
pandas=1.3.5
rdkit=2019.09.3
scikit-learn=0.23.2
seaborn=0.11.1
xgboost=1.3.3
```
We suggest using Anaconda to build the python environment, as there are several default packages used in the scripts but not mentioned above. All test were executed under Ubuntu 18.04.
# Installation guide
The Anaconda can be download in [here](https://www.anaconda.com/products/distribution). The third-party python packages mentioned above can be installed follow the commands below. (The specific version of these packages are recommended to reproduce all results in the paper but not mandatory.)
```
conda create -n py37 python=3.7
conda activate py37
conda install rdkit=2019.09.3 -c rdkit
conda install numpy=1.19.2 scikit-learn=0.23.2 seaborn=0.11.1
pip install ase==3.21.0 dscribe==1.0.0 xgboost==1.3.3
conda install openbabel -c conda-forge
pip install sigopt==7.3.0 ipykernel==5.5.0
python -m ipykernel install --user --name py37 --display-name "Python [conda env:py37"]
```
It takes about 15 minutes to install these third-party python packages in our station.
# Demo & Instructions for use
Here we provide several notebooks to demonstrate how to vectorize the reaction dataset, perform cross-validation, transform the unbalanced target and predict the whole synthetic space using Δ-learning approach. 

[Notebook 1](https://github.com/licheng-xu-echo/SyntheticSpacePrediction/blob/main/notebook/1-Cross-Validation-of-ML-Algorithms-and-Descriptors.ipynb) demonstrates how to use demo data to perform cross-validation of ML algorithms and descriptors. It takes roughly 1 hour 30 minutes to run this notebook in our station (AMD Ryzen 3970X 32-core processor).

[Notebook 2](https://github.com/licheng-xu-echo/SyntheticSpacePrediction/blob/main/notebook/2-Feature-Selection-and-Regression-Performance.ipynb) demonstrates how to perform feature selection at the best combination of descriptor and ML algorithm. With the optimized descriptors, we show regression performance at the out-of-sample test set, which is also shown in Fig. 3b in the [main text](). In addition, the complete feature ranking result is also shown in this notebook, whose top-5 features are displayed in Fig. 3d in the [main text]().  It takes roughly 30 seconds to run this notebook in our station.

[Notebook 3](https://github.com/licheng-xu-echo/SyntheticSpacePrediction/blob/main/notebook/3-Holistic-Prediction-over-Synthetic-Space.ipynb) demonstrates how to use available variables to construct synthetic space and perform prediction over this massive synthetic space. The results in Fig. 4 can be provided with this notebook. Due to file size limitations, we don't provide prediction results of the massive synthetic space of 889056 possibilities directly, but rather provide this notebook to generate these predictions. It takes roughly 20 minutes to run this notebook in our station.

[Notebook 4](https://github.com/licheng-xu-echo/SyntheticSpacePrediction/blob/main/notebook/4-Target-Transformation.ipynb) demonstrates how to perform 4 different transformation methods at unbalanced target. It takes less than 5 seconds to run this notebook in our station.


# How to cite
The paper is under review.
# Contact with us
Email: hxchem@zju.edu.cn; licheng_xu@zju.edu.cn
