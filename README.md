# scDBM-paper
This is the implementation for **"Synthetic Single-Cell RNA-Sequencing Data from Small Pilot Studies using Deep Generative Models"**.

## Abstract  
Deep generative models, such as variational autoencoders (VAEs) or deep Boltzmann machines (DBM), can generate an arbitrary number of synthetic observations after being trained on an initial set of samples. This has mainly been investigated for imaging data but could also be useful for single-cell transcriptomics (scRNA-seq). A small pilot study could be used for planning a full-scale study by investigating planned analysis strategies on synthetic data with different sample sizes. It is unclear whether synthetic observations generated based on a small scRNA-seq dataset reflect the properties relevant for subsequent data analysis steps.\\
We specifically investigated two deep generative modeling approaches, VAEs and DBMs. First, we considered single-cell variational inference (scVI) in two variants, generating samples from the posterior distribution, the standard approach, or the prior distribution. Second, we propose single-cell deep Boltzmann machines (scDBM). When considering the similarity of clustering results on synthetic data to ground-truth clustering, we find that the scVI (posterior) variant resulted in high variability, most likely due to amplifying artifacts of small datasets. All approaches showed mixed results for cell types with different abundance by overrepresenting highly abundant cell types and missing less abundant cell types. With increasing pilot dataset sizes, the proportions of the cells in each cluster became more similar to that of ground-truth data. We also showed that all approaches learn the univariate distribution of most genes, but problems occurred with bimodality. Across all analyses, in comparing 10x Genomics and Smart-seq2 technologies, we could show that for 10x datasets, which have higher sparsity, it is more challenging to make an inference from small to larger datasets. Overall, the results showed that generative deep learning approaches might be valuable for supporting the design of scRNA-seq experiments.

## Data  

The data folder contains an R markdown file with a script to prepare the Segerstolpe data. We also write the gene names and cluster labels into separate files.

* data_preparation.Rmd
* segerstolpe_gene_names.csv
* segerstolpe_hvg.csv
* segerstolpe_hvg_clustering.csv

## Pluto notebooks

The pluto_notebook folder contains Pluto notebooks for the analyses of the Segerstolpe dataset, both to generate the data in the body of the manuscript and generate the supplement's data.

* scDBM_notebook.jl
* supplement_scDBM_notebook.jl

## Plotting  

The plotting folder contains the R markdown scripts to reproduce both the main figures and the supplementary figures.

* figure_2.Rmd
* figure_3.Rmd
* figure_S4.Rmd
* figure_S5.Rmd
* table_1.Rmd

## Main requirements  
Julia: 1.5.0  
scvi: 0.6.0  

## References  

The package is based on the implementation 'BoltzmannMachines.jl'

[1] Lenz, Stefan, Moritz Hess, and Harald Binder. "Unsupervised deep learning on biomedical data with BoltzmannMachines. jl." bioRxiv (2019): 578252.

