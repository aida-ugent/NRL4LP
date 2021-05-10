# Benchmarking Network Embedding Models for Link Prediction: Are We Making Progress?

This repository contains the instructions and materials necessary for reproducing the experiments presented in the 
paper: *Benchmarking Network Embedding Models for Link Prediction: Are We Making Progress?*

The repository is maintained by Alexandru Mara (alexandru.mara@ugent.be).

## Reproducing Experiments
In order to reproduce the experiments presented in the paper the following steps are necessary:

1. Download and install the EvalNE library v0.3.2 as instructed by the authors [here](https://github.com/Dru-Mara/EvalNE)
2. Download and install the implementations of the baseline methods reported in the 
[manuscript](https://arxiv.org/abs/2002.11522). 
We recommend that each method is installed in a unique virtual environment to ensure that the right 
dependencies are used. 
3. Download the datasets used in the experiments: 

    * [StudentDB](http://adrem.ua.ac.be/smurfig)
    * [Facebook](https://snap.stanford.edu/data/egonets-Facebook.html)
    * [BlogCatalog](http://socialcomputing.asu.edu/datasets/BlogCatalog3) 
    * [Flickr](http://socialcomputing.asu.edu/datasets/Flickr)
    * [YouTube](http://socialcomputing.asu.edu/datasets/YouTube2)
    * [GR-QC](https://snap.stanford.edu/data/ca-GrQc.html)
    * [DBLP](https://snap.stanford.edu/data/com-DBLP.html)
    * [PPI](http://snap.stanford.edu/node2vec/#datasets)
    * [Wikipedia](http://snap.stanford.edu/node2vec/#datasets)

4. Modify the `.ini` configuration files from this folder to match the paths where the *datasets* are
stored on your system as well as the paths where the *methods* are installed. Run the evaluation as:

    ```bash
    python -m evalne ./experiments/expLP1.ini
    ```

**NOTE:** In order to obtain the results for, e.g. different values of the embedding dimensionality, the 
conf file `expLP1.ini` has to be modified accordingly and the previous command rerun.

**NOTE:** For AROPE, VERSE and the GEM library, special `main.py` files are required in order to run the 
evaluation through EvalNE. Once these methods are installed, the corresponding main file has to be added 
to the root folder of the method and called from the `.ini` configuration file. These `main.py` files are 
located in a `main_files` folder.


## Citation ##
If you have found our research useful, please consider citing our 
[paper](https://ieeexplore.ieee.org/document/9260030), which is also available on [arxiv](https://arxiv.org/abs/2002.11522):

```bibtex
@INPROCEEDINGS{9260030,
  author={A. C. {Mara} and J. {Lijffijt} and T. d. {Bie}},
  booktitle={2020 IEEE 7th International Conference on Data Science and Advanced Analytics (DSAA)}, 
  title={Benchmarking Network Embedding Models for Link Prediction: Are We Making Progress?}, 
  year={2020},
  volume={},
  number={},
  pages={138-147},
  doi={10.1109/DSAA49011.2020.00026}}
```
