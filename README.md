# SyntheticSpacePrediction
This is a repository for paper *"Holistic Prediction of Synthetic Space by Implementing Transition State Knowledge in Machine Learning"*.
# Introduction
Asymmetric catalysis provides a central strategy for molecular synthesis. Due to the high-dimensional nature of the structure-enantioselectivity relationship, the enantioselectivity prediction in asymmetric catalysis has been a long-standing challenge in synthetic chemistry. The incomprehensive and inaccurate understanding of the synthetic space results in laborious and time-consuming efforts in the discovery of asymmetric reactions, even if the same transformation has already been optimized on model substrates. Here we present a data-driven workflow to achieve holistic prediction of the synthetic space by implementing transition state knowledge in machine learning. Confirmed in the enantioselectivity prediction of asymmetric palladium-catalyzed electro-oxidative C–H bond activation, the vectorization of transition state knowledge allows for excellent descriptive ability and extrapolation capability of the machine learning model, which enables the quantitative evaluation of the massive synthetic space of 846720 possibilities. Model interpretation reveals the non-intuitive olefin effect on the enantioselectivity determination, and subsequent density functional theory calculations elucidates the hidden mechanistic knowledge that the enantioselectivity-determining step depends on the insertion reactivity of the olefin. This synergistic feedback highlights the complementary features of knowledge-based machine learning and interpretation-driven mechanistic study. The established workflow provides the opportunity to harness the buried value of the widely existing catalysis screening data and the transition state models in molecular synthesis. 
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
pytorch=1.9.1
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
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch
pip install sigopt==7.3.0 ipykernel==5.5.0
python -m ipykernel install --user --name py37 --display-name "Python [conda env:py37"]
```
It takes about 20 minutes to install these third-party python packages in our station.
# Demo & Instructions for use
Here we provide several notebooks to demonstrate how to vectorize the reaction data, perform hyperparameter optimization, perform cross-validation, transform the unbalanced target and predict the whole synthetic space using Δ-learning approach. 

[Notebook 1](https://github.com/licheng-xu-echo/SyntheticSpacePrediction/blob/main/1-Descriptor-Generation.ipynb) demonstrates how to generate the widely used descriptors which are used in the benchmark procedure. It takes within 1 minute in our station (Intel(R) Core(TM) i9-9900K CPU).

[Notebook 2](https://github.com/licheng-xu-echo/SyntheticSpacePrediction/blob/main/2-Target-Transformation.ipynb) demonstrates how to perform 4 different transformation methods at unbalanced target. It takes less than 5 seconds to run this notebook in our station.

[Notebook 3](https://github.com/licheng-xu-echo/SyntheticSpacePrediction/blob/main/3-Hyperparameter-Optimization-and-Benchmark.ipynb) demonstrates how to perform hyperparameter optimization for every machine learning model and compare the model performance of different combination of descriptor and machine learning model. It takes about 75 minutes to run this notebook in our station.

[Notebook 4](https://github.com/licheng-xu-echo/SyntheticSpacePrediction/blob/main/4-Feature-Selection-and-Regression-Performance.ipynb) demonstrates how to perform feature selection on our PhysOrg descriptors, select out-of-sample test set and show the model performance at 10-fold cross-validation task and 2 extrapolative task. It takes less than 5 minutes to run this notebook in our station.

[Notebook 5](https://github.com/licheng-xu-echo/SyntheticSpacePrediction/blob/main/5-Holistic-Prediction-of-Synthetic-Space.ipynb) demonstrates how to use available variables to construct synthetic space and perform prediction over this massive synthetic space. The results in Fig. 6 can be provided with this notebook. Due to file size limitations, we don't provide prediction results of the massive synthetic space of 846720 possibilities directly, but rather provide this notebook to generate these predictions. It takes roughly 1 hour to run this notebook in our station.

[Notebook 6](https://github.com/licheng-xu-echo/SyntheticSpacePrediction/blob/main/6-Comparison-of-Different-Vectorization-Approach.ipynb) shows the model performance of different vectorization approaches (with or without transition state information). It takes roughly 10 minutes to run this notebook in our station.

# Check list of compounds in the paper
A check list table is provided to match the compound indices in the paper and the indices of *.xyz* files in the repository which are used to generate descriptors.

![compoundlist](/image/2.png)

| Compound index in the paper | File index in the repository| | Compound index in the paper | File index in the repository|
| :----: | :----: | :----: | :----: | :----: |
| **1** | imine-19 (aldehyde-19) | | **2** | imine-16 (aldehyde-16) |
| **3** | imine-15 (aldehyde-15) | | **4** | imine-8 (aldehyde-8) |
| **5** | imine-2 (aldehyde-2) | | **6** | imine-1 (aldehyde-1) |
| **7** | imine-6 (aldehyde-6) | | **8** | imine-7 (aldehyde-7) |
| **9** | imine-10 (aldehyde-10) | | **10** | imine-14 (aldehyde-14) |
| **11** | imine-18 (aldehyde-18) | | **12** | imine-17 (aldehyde-17) |
| **13** | olefin-8 | | **14** | olefin-28 |
| **15** | olefin-23 | | **16** | olefin-21 |
| **17** | olefin-27 | | **18** | olefin-20 |
| **19** | olefin-22 | | **20** | olefin-1 |
| **21** | olefin-2 | | **22** | olefin-5 |
| **23** | olefin-13 | | **24** | olefin-14 |
| **25** | olefin-15 | | **26** | olefin-16 |
| **27** | olefin-17 | | **28** | olefin-19 |
| **29** | olefin-11 | | **30** | olefin-24 |
| **31** | olefin-25 | | **32** | olefin-26 |
| **33** | olefin-29 | | **34** | Pd-complex-2 (TDG-2) |
| **35** | Pd-complex-20  (TDG-20) | | **36** | Pd-complex-21  (TDG-21) |
| **37** | Pd-complex-3  (TDG-3) | | **38** | Pd-complex-4  (TDG-4) |
| **39** | Pd-complex-7  (TDG-7) | | **40** | Pd-complex-8  (TDG-8) |
| **41** | Pd-complex-13  (TDG-13) | | **42** | Pd-complex-12  (TDG-12) |
| **43** | Pd-complex-16  (TDG-16) | | **44** | Pd-complex-6  (TDG-6) |
| **45** | Pd-complex-17  (TDG-17) | | **46** | Pd-complex-5  (TDG-5) |
| **47** | Pd-complex-19  (TDG-19) | | **48** | Pd-complex-14  (TDG-14) |
| **49** | Pd-complex-15  (TDG-15) | | **50** | Pd-complex-0  (TDG-0) |
| **51** | Pd-complex-10  (TDG-10) | | **52** | Pd-complex-1  (TDG-1) |
| **53** | Pd-complex-9  (TDG-9) | |        |         |
| **68 (new)** | imine-4  (aldehyde-4) | | **69 (new)** | imine-11  (aldehyde-11) |
| **70 (new)** | imine-9  (aldehyde-9) | | **71 (new)** | imine-13  (aldehyde-13) |
| **72 (new)** | olefin-4 | | **73 (new)** | olefin-12 |
| **74 (new)** | olefin-0 | | **75 (new)** | olefin-30 |
| **76 (new)** | Pd-complex-18  (TDG-18) | | **77 (new)** | Pd-complex-11  (TDG-11) |

# How to cite
The paper is under review.
# Contact with us
Email: hxchem@zju.edu.cn; licheng_xu@zju.edu.cn; angellasty@zju.edu.cn
