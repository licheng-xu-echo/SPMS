# SPMS
This is a repository for paper: [*A Molecular Stereostructure Descriptor based on Spherical Projection*](https://www.thieme-connect.de/products/ejournals/abstract/10.1055/s-0040-1705977)

## Introduction
Description of molecular stereostructure is critical for the machine learning prediction of asymmetric catalysis. We develop a spherical projection descriptor of molecular stereostructure (SPMS), which allows precise representation of the molecular vdW surface. 

This project provides the key script to generate SPMS based on MDL SDF files (.sdf, **V2000** version). In addition, we provide two Jupyter Notebooks in **Example** folder to demonstrate how to generate SPMS from SDF files and how to use it for machine learning application on the dataset of asymmetric thiol addition to N-acylimines from Denmark's recent work. ([*Science* **2019**, *363*, eaau5631.](https://science.sciencemag.org/content/363/6424/eaau5631))

This work is published at [*Synlett*](http://doi.org/10.1055/s-0040-1705977). If this project was used in your work, please cite this [paper](http://doi.org/10.1055/s-0040-1705977).

## Dependences

All third-party python packages required for generating SPMS are just [numpy](https://numpy.org/) and [rdkit](http://rdkit.org/).

In order to run Jupyter Notebook for machine learning application demonstration, several machine learning, deep learning and visualazation third-party python packages are required.

```
python>=3.6
numpy>=1.17.4
rdkit>=2019.03.2
pandas>=1.1.1
tensorflow-gpu=1.14.0
scikit-learn>=0.22
seaborn>=0.9.0
```
The version of dependent third-party packages above are recommended. The virsion of TensorFlow should be **1.X**. We suggest using [Anaconda](https://www.anaconda.com/) to install python3.6 or higher version, as conda and pip together make the installation of these dependences much easier.

## Usage
The core script to generate SPMS is [SPMS.py](https://github.com/licheng-xu-echo/SPMS/blob/master/SPMS.py), which is very easy to understand and use for its few code lines.

There is an example file [L-proline](https://github.com/licheng-xu-echo/SPMS/blob/master/Example/sdf_examples/L-proline.sdf) for demonstration. The chiral carbon is selected as the key atom which will be placed at the origin of cartesian coordinate system. Several key atoms to generate SPMS is also supported. If key atom is not defined, **the center of  mass** will be placed at the origin of cartesian coordinate system. The resolution of SPMS is controlled by **desc_n** and **desc_m**. If **sphere_radius** paramter is not set, the smallest radius to hold the whole molecule will be calulated and used. The simplest usage is just like below:
```
from SPMS import SPMS
## Initiaze the SPMS
spms = SPMS('./L-proline.sdf',key_atom_mum=[3],desc_n=40,desc_m=40,sphere_radius=8)

## Calculate the SPMS
spms.GetSphereDescriptors()
desc = spms.sphere_descriptors
```

More details for the usage of SPMS and machine learning application of SPMS, please check two Jupyter Notebooks in **Example** folder.

We also provide a [website](http://www.spmsgen.net/) for **drawing SPMS figures** for chemical interpretation.

## How to cite
If the method is used in your paper, please cite as: *Synlett* **2021**, *32*, 1837.

## Contact us
Email: hxchem@zju.edu.cn; licheng_xu@zju.edu.cn
