# SALT
Codes for WACVW paper 'Small or Far Away? Exploiting Deep Super-Resolution and Altitude Data for Real-World Aerial Animal Surveillance'

This repository contains the source code that accompanies our paper "Small or Far Away? Exploiting Deep Super-Resolution and Altitude Data for Real-World Aerial Animal Surveillance". The paper is available at https://arxiv.org/abs/2111.06830.

Useage
-----
First download the dataset at https://zenodo.org/record/1204408#.YdDIQGjP1PZ. 
Then use the code in the 'pre-process' folder to till the big images into patches and extract the altitude information from the image.
Then use 'main.py' to train the network or use 'detect.py' to test the network.

Citation
---------
Consider citing our work in your own research if this repository is useful:
>@article{DBLP:journals/corr/abs-2111-06830,
>  author    = {Mowen Xue and
>               Theo Greenslade and
>               Majid Mirmehdi and
>               Tilo Burghardt},
>  title     = {Small or Far Away? Exploiting Deep Super-Resolution and Altitude Data
               for Aerial Animal Surveillance},
>  journal   = {CoRR},
>  volume    = {abs/2111.06830},
>  year      = {2021},
>  url       = {https://arxiv.org/abs/2111.06830},
>  eprinttype = {arXiv},
>  eprint    = {2111.06830},
>  timestamp = {Tue, 16 Nov 2021 12:12:31 +0100},
>  biburl    = {https://dblp.org/rec/journals/corr/abs-2111-06830.bib},
>  bibsource = {dblp computer science bibliography, https://dblp.org}
>}
