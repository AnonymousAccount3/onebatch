# One Batch Greedy K-medoids

This repository provides the results of the experiements conducted on several UCI datasets.

Please clone the repository and install the dependencies with:
```
pip install -r requirements.txt
```
To run the experiments for the config file `configs/expe_large_scale.yml` use the following command line:
```
python script.py --config expe_large_scale
```

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
 - [BanditPAM++](https://arxiv.org/abs/2310.18844) "BanditPAM++: Faster k-medoids Clustering". The number of SWAP iteration is taken in [0, 2, 5]
 - OneBatch: The batch size is taken in [100, 300, 500, 1000]

## Results Summary

This section presents the summary of the results. Relative Time (RT) and Delta Relative Objective ($\\Delta$RO), averaged over the results for $K = [10, 50, 100]$, are reported in the following tables. The Relative Time and the Delta Relative Objective are respectively defined as follows:

$$ \\text{RT}(A) = T(A) / T(A^*) $$

$$ \\Delta\\text{RO}(A) = \\text{Obj}(A) / \\text{Obj}(A^*) - 1$$

Where $T(A)$ and $\\text{Obj}(A)$ are respectively the computational time and the objective of algorithm $A$, and $A^*$ is the algorithm of lowest objective.

Note: BanditPAM++ is not included in the summary as it reaches the limit of computational time for $K$ in [50, 100]. ([See Detailed Results below](https://github.com/AnonymousAccount3/onebatch/tree/main?tab=readme-ov-file#detailed-results)).

### Average Delta RO
| method         |   covertype |   dota2 |   monitor_gas |   optical_radar |   phishing |
|:---------------|------------:|--------:|--------------:|----------------:|-----------:|
| Random         |        40   |     3   |          21.5 |            17.6 |       19   |
| KMC2-20        |        27.5 |     2.9 |          17.8 |            18.5 |       21.2 |
| KMC2-100       |        26.3 |     2.9 |          17.6 |            18.2 |       22.1 |
| KMC2-200       |        29.4 |     2.9 |          18.6 |            18.3 |       20.7 |
| KMeans++       |        27.9 |     2.9 |          20.6 |            17.5 |       22.4 |
| LS-KMeans++-5  |        22.2 |     2.6 |          16.1 |            15.1 |       19.4 |
| LS-KMeans++-10 |        19.1 |     2.4 |          12.8 |            13.5 |       17.2 |
| FastCLARA-5    |         8.5 |     1.8 |           4.4 |             8.3 |        5.2 |
| FastCLARA-50   |         6.2 |     1.6 |           3.2 |             7.1 |        4.3 |
| OneBatch-100   |        24.6 |     1.5 |          12.7 |             7.9 |       11.5 |
| OneBatch-300   |         8.5 |     0.9 |           4.9 |             3.1 |        4.9 |
| OneBatch-500   |         3.1 |     0.6 |           2.5 |             1.7 |        2.5 |
| OneBatch-1000  |         0   |     0.2 |           0.3 |             0   |        0.3 |

### Average RT
| method         |   covertype |   dota2 |   monitor_gas |   optical_radar |   phishing |
|:---------------|------------:|--------:|--------------:|----------------:|-----------:|
| Random         |         0   |     0   |           0   |             0.1 |        0   |
| KMC2-20        |         0.7 |     4.1 |           1   |             2   |        1.4 |
| KMC2-100       |         3.4 |    23.1 |           5.1 |             9.7 |        7.5 |
| KMC2-200       |         7.9 |    47.7 |          11.2 |            19.2 |       15.3 |
| KMeans++       |        16.4 |    34.1 |          11   |            90.9 |       14.4 |
| LS-KMeans++-5  |        85.3 |    69.6 |          91.3 |           167.1 |       79.3 |
| LS-KMeans++-10 |       152.1 |   109.5 |         172   |           246.1 |      127.2 |
| FastCLARA-5    |        18.2 |    14.7 |          10.2 |            32.5 |       12.1 |
| FastCLARA-50   |       179.9 |   128.1 |          99.6 |           321.2 |      120   |
| OneBatch-100   |        13   |     9.3 |           7.4 |            12.5 |        8.5 |
| OneBatch-300   |        29.6 |    26.5 |          20   |            26.9 |       23   |
| OneBatch-500   |        48.9 |    40.2 |          33.1 |            35.3 |       34.5 |
| OneBatch-1000  |       100   |    67.3 |          67.8 |            69.2 |       68.5 |

## Detailed Results

![plot](/figures/dota2_rt_ro.png)

![plot](/figures/phishing_rt_ro.png)

![plot](/figures/optical_radar_rt_ro.png)

![plot](/figures/monitor_gas_rt_ro.png)

![plot](/figures/covertype_rt_ro.png)

## Time vs Objective

The following figures present a visualization of the pairs ($T(A)$, Obj($A$)) for each algorithm $A$ in 2D graphics. The red line is the Pareto front. Each method belonging to the Pareto front is the best method for at least one Time vs Obj trade-off.

![plot](/figures/dota2_10_time_vs_obj.png) 

![plot](/figures/dota2_100_time_vs_obj.png)

![plot](/figures/phishing_10_time_vs_obj.png) 

![plot](/figures/phishing_100_time_vs_obj.png)

![plot](/figures/optical_radar_10_time_vs_obj.png) 

![plot](/figures/optical_radar_100_time_vs_obj.png)

![plot](/figures/monitor_gas_10_time_vs_obj.png) 

![plot](/figures/monitor_gas_100_time_vs_obj.png)

![plot](/figures/covertype_10_time_vs_obj.png) 

![plot](/figures/covertype_100_time_vs_obj.png)


## Testing Assumptions

This section provides the test for the three assumptions required to derive the OneBatch time complexity. $G_{m_k}$ is the gain row of the selected medoid at step $k$. The "Col sparisty ratio" in red is the mean sparisty of the columns of $G$ with non null value in the $m_k$ row. We observe that for all datasets, the three sparsity ratios are below $1/k$. The assumptions are then verfied for these datasets. The results can be reproduce through the ``Sparsity.ipynb`` notebook.

![plot](/figures/sparsity_mnist.png)

![plot](/figures/sparsity_cifar.png) 

![plot](/figures/sparsity_dota2.png) 

![plot](/figures/sparsity_phishing.png) 

![plot](/figures/sparsity_optical_radar.png) 

![plot](/figures/sparsity_monitor_gas.png) 

![plot](/figures/sparsity_covertype.png) 
