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
 - Random: Select medoids uniformly at random
 - [KMC2](https://ojs.aaai.org/index.php/AAAI/article/view/10259/10118): "Approximate k-means++ in sublinear time". The chain length parameter is taken in [20, 100, 200]
 - [KMeans++](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) "k-means++: The Advantages of Careful Seeding"
 - [LS-KMeans++](https://proceedings.mlr.press/v97/lattanzi19a/lattanzi19a.pdf) "A better k-means++ algorithm via local search". The number of local search loops is taken in [5, 10]
 - [FastCLARA](https://www.sciencedirect.com/science/article/pii/S0306437921000557) "Fast and eager medoids clustering: runtime improvement of the PAM, CLARA, and CLARANS algorithms". The number of iteration is taken in [5, 50]
 - OneBatch: The batch size is taken in [100, 300, 500, 1000]

## Results Summary

This section presents the summary of the results. Relative Time (RT) and Delta Relative Objective ($\\Delta$RO), averaged over the results for $K = [10, 50, 100]$, are reported in the following tables. The Relative Time and the Delta Relative Objective are respectively defined as follows:

$$ \\text{RT}(A) = T(A) / T(A^*) $$

$$ \\Delta\\text{RO}(A) = \\text{Obj}(A) / \\text{Obj}(A^*) - 1$$

Where $T(A)$ and $\\text{Obj}(A)$ are respectively the computational time and the objective of algorithm $A$, and $A^*$ is the algorithm of lowest objective.

### Average Delta RO
| method         |   covertype |   dota2 |   monitor_gas |   optical_radar |   phishing |
|:---------------|------------:|--------:|--------------:|----------------:|-----------:|
| Random         |        40   |     2.8 |          21.2 |            17.6 |       18.7 |
| KMC2-20        |        27.5 |     2.7 |          17.5 |            18.5 |       20.8 |
| KMC2-100       |        26.3 |     2.7 |          17.4 |            18.1 |       21.8 |
| KMC2-200       |        29.4 |     2.7 |          18.3 |            18.3 |       20.4 |
| KMeans++       |        27.9 |     2.7 |          20.3 |            17.5 |       22.1 |
| LS-KMeans++-5  |        22.2 |     2.4 |          15.9 |            15   |       19.1 |
| LS-KMeans++-10 |        19.1 |     2.2 |          12.5 |            13.5 |       16.9 |
| FastCLARA-5    |         8.5 |     1.6 |           4.2 |             8.3 |        4.9 |
| FastCLARA-50   |         6.2 |     1.4 |           3   |             7.1 |        4   |
| OneBatch-100   |        24.6 |     1.3 |          12.5 |             7.9 |       11.2 |
| OneBatch-300   |         8.5 |     0.7 |           4.7 |             3.1 |        4.6 |
| OneBatch-500   |         3.1 |     0.4 |           2.3 |             1.6 |        2.2 |
| OneBatch-1000  |         0   |     0   |           0.1 |             0   |        0   |

### Average RT
| method         |   covertype |   dota2 |   monitor_gas |   optical_radar |   phishing |
|:---------------|------------:|--------:|--------------:|----------------:|-----------:|
| Random         |         0   |     0   |           0   |             0.1 |        0.1 |
| KMC2-20        |         0.7 |     4.3 |           1.2 |             2.1 |        1.5 |
| KMC2-100       |         3.4 |    24.1 |           5.7 |            10   |        8.2 |
| KMC2-200       |         7.9 |    49.7 |          12.4 |            19.9 |       16.7 |
| KMeans++       |        16.4 |    35.9 |          12.3 |            96   |       16.2 |
| LS-KMeans++-5  |        85.3 |    72.6 |          94.8 |           175.9 |       83.5 |
| LS-KMeans++-10 |       152.1 |   113.8 |         177.8 |           258.3 |      133.2 |
| FastCLARA-5    |        18.2 |    17.3 |          11.4 |            40.2 |       15.3 |
| FastCLARA-50   |       179.9 |   151   |         113.2 |           397.5 |      151.7 |
| OneBatch-100   |        13   |    14   |          14.5 |            18   |       16.5 |
| OneBatch-300   |        29.6 |    39.4 |          39.8 |            39.1 |       44.8 |
| OneBatch-500   |        48.9 |    59.2 |          65.9 |            51.1 |       66.9 |
| OneBatch-1000  |       100   |   100   |         135   |           100   |      132.8 |

## Detailed Results



![plot](/figures/dota2_100_time_vs_obj.png) 

![plot](/figures/dota2_100_time_vs_obj.png)
