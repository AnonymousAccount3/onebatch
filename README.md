# One Batch Greedy K-medoids

This repository provides the results of the experiements conducted on several UCI datasets.

## Datasets

The five following datasets are used:
 - ``dota2`` Shape: (92650, 117) [[description](https://archive.ics.uci.edu/dataset/367/dota2+games+results
)] [[download](https://archive.ics.uci.edu/static/public/367/dota2+games+results.zip)] 
 - ``phishing`` Shape: (235795, 51) [[description](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)]  [[download](https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip)]
 - ``optical_radar`` Shape: (325834, 175) [[description](https://archive.ics.uci.edu/dataset/525/crop+mapping+using+fused+optical+radar+data+set)] [[download](https://archive.ics.uci.edu/static/public/525/crop+mapping+using+fused+optical+radar+data+set.zip)]
 - ``monitor_gas`` Shape: (416153, 9) [[description](https://archive.ics.uci.edu/dataset/799/single+elder+home+monitoring+gas+and+position)] [[download](https://archive.ics.uci.edu/static/public/799/single+elder+home+monitoring+gas+and+position.zip)]
 - ``covertype`` Shape: (581011, 55) [[description](https://archive.ics.uci.edu/dataset/31/covertype)] [[download](https://archive.ics.uci.edu/static/public/31/covertype.zip)]

To reproduce the experiements, please download the datasets and unzip them in a ``./data/`` folder.

## Competitors

The following competitors are considered:
 - Random: Select uniformly at random
 - [KMC2](https://ojs.aaai.org/index.php/AAAI/article/view/10259/10118): "Approximate k-means++ in sublinear time". The chain length parameter is taken in [20, 100, 200]
 - [KMeans++](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) "k-means++: The Advantages of Careful Seeding"
 - [LS-KMeans++](https://proceedings.mlr.press/v97/lattanzi19a/lattanzi19a.pdf) "A better k-means++ algorithm via local search". The number of local search loops is taken in [5, 10]
 - [FastCLARA](https://www.sciencedirect.com/science/article/pii/S0306437921000557) "Fast and eager medoids clustering: runtime improvement of the PAM, CLARA, and CLARANS algorithms". The number of iteration is taken in [5, 50]
 - OneBatch: The batch size is taken in [100, 300, 500, 1000]

## Results Summary

## Plots

![plot](/figures/dota2_100_time_vs_obj.png) 

![plot](/figures/dota2_100_time_vs_obj.png)
