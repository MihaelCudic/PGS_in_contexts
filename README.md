# Code for the analysis in the manuscript titled:
# "Putting Polygenic Scores in Context: How Intersectional Factors Affect Relative and Absolute Genetic Risk

Authors: Mihael Cudic1,2,3 ; Justin D. Tubbs1,2,3 ; Tian Ge1,2,3 ; Jordan W. Smoller1,2,3 

1 Psychiatric and Neurodevelopmental Genetics Unit, Center for Genomic Medicine, Massachusetts General Hospital, Boston, MA
2 Center for Precision Psychiatry, Department of Psychiatry, Massachusetts General Hospital, Boston, MA
3 Stanley Center for Psychiatric Research, Broad Institute of MIT and Harvard, Cambridge, MA

### 1. Enter directories for PGS and environmental covariates in options/base_options.yaml
      prs_file:  # Insert file location to PGS
      env_file:  # Insert file location to covariates

### 2. main*.ipynb denote the jupyter notebooks to run and generate figures from the main analysis
      main_00: Code to generate Figure 1 of manuscript
      main_01: Code to run the main analysis of manuscript
            - OR and AR estimates per one-way, two-way, and three-way intersections
            - Max. OR % difference across variable and two-way variable intersections
            - Boostrapping of results and subsequent calculation of emperical p-values
      main_02: Plot heatmaps
            - OR and ARs estiamtes across one-way and two-way intersections
            - Max OR % difference across one-way and two-way intersections
      main_03: Plot max OR % difference for each phenotype acros one-way, two-way and three-way intersections
      main_04: Code to produce figures comparing estimates across cohorts (i.e. UKB Eur., AoU Eur., and AoU Afr.)
            
### 3. supp*.ipynb denote the jupyter notebooks to run ang generate figures from the supplemental analysis
      supp_00: A visualization of the trends between household income and deprivation
      supp_01a: Analysis of Pain et al. sensitivity to nongaussian PGS distrbutions
      supp_01b: Analysis of Pain et al. sensitivity to nonuniform PGS effect sizes
      supp_2a: Perform logistic regression to compute top OR as alternative to Pain et al.
      supp_2b: Plot the OR estimates from Pain et al. and logistic regression (supp_2a)
      supp_3a: Run logisitic regression for interaction terms as a sensivity analysis
      supp_3b: Compare p-values obtained in main analysis with that of supp_3a
      supp_04a: Generate simulated data for sensitivity analysis
      supp_04b: Run main analysis on simulated data from supp_04a
      supp_04c: Perform random permutation of phenotype label as a sensitivity analysis
      supp_4d: Plot the results from the simulation and random permutation sensitivity analysis
