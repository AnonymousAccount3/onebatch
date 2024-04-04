# Additional Small-scale Experiments

This sub-directory provides the results of the experiements conducted on several "small-scale" UCI datasets.

To run the experiments for the config file `./configs/expe_small_scale.yml` use the following command line:
```
python script.py --config expe_small_scale
```

## Datasets

The five following datasets are used:
 - ``abalone`` Shape: (4176, 8) [[description](https://archive.ics.uci.edu/dataset/1/abalone)] [[download](https://archive.ics.uci.edu/static/public/1/abalone.zip)] 
 - ``wine`` Shape: (4898, 12) [[description](https://archive.ics.uci.edu/dataset/186/wine+quality)]  [[download](https://archive.ics.uci.edu/static/public/186/wine+quality.zip)]
 - ``bankruptcy`` Shape: (6819, 96) [[description](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction)] [[download](https://archive.ics.uci.edu/static/public/572/taiwanese+bankruptcy+prediction.zip)]
 - ``drybean`` Shape: (13611, 16) [[description](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)] [[download](https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip)]
 - ``letter`` Shape: (19999, 16) [[description](https://archive.ics.uci.edu/dataset/59/letter+recognition)] [[download](https://archive.ics.uci.edu/static/public/59/letter+recognition.zip)]

To reproduce the experiements, please download the datasets and unzip them in a ``./data/`` folder.

## Competitors

The following competitors are considered:
 - **Random**: Select medoids uniformly at random
 - [KMC2](https://ojs.aaai.org/index.php/AAAI/article/view/10259/10118): "Approximate k-means++ in sublinear time". The chain length parameter is taken in [20, 100, 200]
 - [KMeans++](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) "k-means++: The Advantages of Careful Seeding"
 - [LS-KMeans++](https://proceedings.mlr.press/v97/lattanzi19a/lattanzi19a.pdf) "A better k-means++ algorithm via local search". The number of local search loops is taken in [5, 10]
 - [FastCLARA](https://www.sciencedirect.com/science/article/pii/S0306437921000557) "Fast and eager medoids clustering: runtime improvement of the PAM, CLARA, and CLARANS algorithms". The number of iteration is taken in [5, 50]
 - [PAM variants](https://www.sciencedirect.com/science/article/pii/S0306437921000557) "Fast and eager medoids clustering: runtime improvement of the PAM, CLARA, and CLARANS algorithms". The implementation of the methods is provided in the following Python library: [kmedoids](https://github.com/kno10/python-kmedoids). The following variants are considered [[fastpam1](https://python-kmedoids.readthedocs.io/en/latest/#fastpam1), [fasterpam](https://python-kmedoids.readthedocs.io/en/latest/#fasterpam), [alternate](https://python-kmedoids.readthedocs.io/en/latest/#alternating-k-medoids-k-means-style)]
 - **Greedy variants**: The implementation of the methods is provided in the following Python library: [apricot](https://github.com/jmschrei/apricot). The following variants are considered [[naive](https://apricot-select.readthedocs.io/en/latest/optimizers/naive.html), [lazy](https://apricot-select.readthedocs.io/en/latest/optimizers/lazy.html), [sample](https://apricot-select.readthedocs.io/en/latest/optimizers/sample.html)]
 - **OneBatch**: The batch size is taken in [100, 300, 500, 1000]

## Results Summary

This section presents the summary of the results. Relative Time (RT) and Delta Relative Objective ($\\Delta$RO), averaged over the results for $K = [10, 50, 100]$, are reported in the following tables. The Relative Time and the Delta Relative Objective are respectively defined as follows:

$$ \\text{RT}(A) = T(A) / T(A^*) $$

$$ \\Delta\\text{RO}(A) = \\text{Obj}(A) / \\text{Obj}(A^*) - 1$$

Where $T(A)$ and $\\text{Obj}(A)$ are respectively the computational time and the objective of algorithm $A$, and $A^*$ is the algorithm of lowest objective.

### Average Delta RO
| method         |   abalone |   bankruptcy |   drybean |   letter |   wine |
|:---------------|----------:|-------------:|----------:|---------:|-------:|
| Random         |      31.9 |         22.5 |      40   |     22.9 |   18.6 |
| KMC2-20        |      26.2 |         23.1 |      27.3 |     21.3 |   19.2 |
| KMC2-100       |      27.1 |         23.5 |      27.8 |     22.5 |   19.2 |
| KMC2-200       |      24.5 |         22.7 |      26.2 |     22.3 |   18.6 |
| KMeans++       |      25.8 |         22.5 |      27.1 |     23.8 |   18   |
| LS-KMeans++-5  |      20.4 |         19.9 |      21.3 |     21.3 |   15.7 |
| LS-KMeans++-10 |      17.8 |         18.3 |      18.6 |     19.7 |   14.2 |
| FastCLARA-5    |       9.8 |         10.4 |      11.5 |     11.1 |    8.4 |
| FastCLARA-50   |       8.5 |          9.2 |       9.6 |     10   |    7.3 |
| Greedy-lazy    |       4   |          2   |       4.9 |      3.2 |    3.4 |
| Greedy-naive   |       4   |          2   |       4.9 |      2.5 |    1.9 |
| Greedy-sample  |       4.1 |          2   |       4.8 |      3.1 |    3.1 |
| PAM-alternate  |       6   |          6.4 |       7.4 |      6.6 |    7.1 |
| PAM-fasterpam  |       0   |          0   |       0   |      0   |    0   |
| PAM-fastpam1   |       0.2 |          0   |       0.1 |      0   |    0   |
| OneBatch-100   |      14.9 |         11.5 |      19   |     14.3 |   11.3 |
| OneBatch-300   |       7.1 |          6.4 |      10.4 |      8.4 |    7.1 |
| OneBatch-500   |       5.3 |          4.4 |       8.3 |      6.1 |    4.9 |
| OneBatch-1000  |       3.9 |          3   |       6.4 |      4.2 |    3   |

### Average RT
| method         |   abalone |   bankruptcy |   drybean |   letter |   wine |
|:---------------|----------:|-------------:|----------:|---------:|-------:|
| Random         |       0   |          0   |       0   |      0   |    0   |
| KMC2-20        |      26.4 |         14.3 |       5.9 |      3   |   42   |
| KMC2-100       |     125.3 |         67.9 |      28.4 |     15   |  200.9 |
| KMC2-200       |     253.9 |        166.3 |      62.3 |     29.8 |  472.6 |
| KMeans++       |       2.7 |          8.1 |       2.7 |      1.8 |    9.2 |
| LS-KMeans++-5  |       9.8 |         24.5 |      16.9 |     13.4 |   37   |
| LS-KMeans++-10 |      16.7 |         40.1 |      31.7 |     25.2 |   64.6 |
| FastCLARA-5    |       3.8 |          2.8 |       1.5 |      1.2 |    5.5 |
| FastCLARA-50   |      31.6 |         63.7 |      24.2 |     18.1 |   73.3 |
| Greedy-lazy    |     518.4 |        167.5 |     223.1 |    161.9 |  597.8 |
| Greedy-naive   |     580.4 |        233.7 |     290.8 |    254.9 |  635.4 |
| Greedy-sample  |     671.5 |        208.8 |     249.4 |    172.4 |  679.4 |
| PAM-alternate  |      79.6 |         61.8 |     102.4 |    109   |  136.6 |
| PAM-fasterpam  |      72.2 |         59.7 |     100   |     77.1 |  100   |
| PAM-fastpam1   |     285.2 |        286.8 |     684.5 |    556.3 |  442   |
| OneBatch-100   |      27   |          6.8 |       4.2 |      1.8 |   25.1 |
| OneBatch-300   |      34.9 |         11.3 |       7.7 |      3.8 |   32.9 |
| OneBatch-500   |      43.7 |         15.7 |      10.5 |      5.2 |   42   |
| OneBatch-1000  |      69.1 |         27.3 |      19.6 |     10.1 |   66.3 |

## Detailed Results

![plot](/figures/abalone_rt_ro.png)

![plot](/figures/wine_rt_ro.png)

![plot](/figures/bankruptcy_rt_ro.png)

![plot](/figures/drybean_rt_ro.png)

![plot](/figures/letter_rt_ro.png)

## Time vs Objective

The following figures present a visualization of the pairs ($T(A)$, Obj($A$)) for each algorithm $A$ in 2D graphics. The red line is the Pareto front. Each method belonging to the Pareto front is the best method for at least one Time vs Obj trade-off.

![plot](/figures/abalone_10_time_vs_obj.png) 

![plot](/figures/abalone_100_time_vs_obj.png)

![plot](/figures/wine_10_time_vs_obj.png) 

![plot](/figures/wine_100_time_vs_obj.png)

![plot](/figures/bankruptcy_10_time_vs_obj.png) 

![plot](/figures/bankruptcy_100_time_vs_obj.png)

![plot](/figures/drybean_10_time_vs_obj.png) 

![plot](/figures/drybean_100_time_vs_obj.png)

![plot](/figures/letter_10_time_vs_obj.png) 

![plot](/figures/letter_100_time_vs_obj.png)


## Testing Assumptions

This section provides the test for the three assumptions required to derive the OneBatch time complexity. $G_{m_k}$ is the gain row of the selected medoid at step $k$. The "Col sparisty ratio" in red is the mean sparisty of the columns of $G$ with non null value in the $m_k$ row. We observe that for all datasets, the three sparsity ratios are below $1/k$. The assumptions are then verfied for these datasets. The results can be reproduce through the ``Sparsity.ipynb`` notebook.

![plot](/figures/sparsity_abalone.png) 

![plot](/figures/sparsity_wine.png) 

![plot](/figures/sparsity_bankruptcy.png) 

![plot](/figures/sparsity_drybean.png) 

![plot](/figures/sparsity_letter.png) 
