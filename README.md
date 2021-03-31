# brittnu: Britt's nu reliability coefficient
Computes Britt's nu as described in Britt (under review). This statistic is designed to assess the reliability of Dirichlet-distributed data. It is especially useful for assessing the reliability of topics extracted via latent Dirichlet allocation (LDA). The allocations of words to topics or of topics to documents may be assessed using Britt's nu.



## Installation

To use this package, first install and load it in R with

```r
install.packages("devtools")
library(devtools)
install_github("bcbritt/brittnu")
library("brittnu")
```



## Purpose

Britt's nu is a family of reliability coefficients used to assess Dirichlet-distributed data. There are four forms of Britt's nu that assess reliability for individual Dirichlet distributions, reliability for sets of multiple Dirichlet distributions, the pairwise reliability for pairs of cross-validation iterations, and the joint reliability across all cross-validation iterations:

- Britt's single-distribution nu for joint reliability
- Britt's single-distribution nu for pairwise reliability
- Britt's omnibus (multiple-distribution) nu for joint reliability
- Britt's omnibus (multiple-distribution) nu for pairwise reliability

Each of these statistics is constructed using similar underlying logic as other interrater reliability coefficients. As such, the family of Britt's nu statistics is particularly well-suited for assessing the reliability of allocations of words to topics or of topics to documents across multiple LDA cross-validations.



## Usage

The `brittnu` function generates all four forms of Britt's nu. This function requires a list of vectors of Dirichlet-distributed observations (`x`). Each list element must be the result of a call to [**topicmodels**::LDA](https://www.rdocumentation.org/packages/topicmodels/versions/0.1-4/topics/LDA) or a 2D double vector with distributions as the rows and each category within each distribution as the columns (such that each row in the vector sums to 1). You should also specify `type="wt"` to assess word-topic reliability, `type="td"` for topic-document reliability, or `type=NA` if `x` consists of raw data rather than results from [**topicmodels**::LDA](https://www.rdocumentation.org/packages/topicmodels/versions/0.1-4/topics/LDA). If the concentration parameters of the Dirichlet distribution are known, you should set `alpha` to a numeric vector of those concentration parameters. Otherwise, if `alpha=NA`, the concentration parameters will be estimated from the data.

The topics extracted via latent Dirichlet allocation may appear in a different order (or be "shuffled") for each cross-validation iteration, so they may need to be reordered before reliability has been assessed. As such, if you are working with LDA results or with other Dirichlet-distributed data in which the categories may not be consistently ordered, you should set `shuffle=TRUE`. Otherwise, leave `shuffle=FALSE`. Likewise, many contexts, including LDA, often assume that the underlying Dirichlet distribution is symmetrical, such that all of the concentration parameters are identical. In such cases, you should set `symmetric_alpha=TRUE`. If this restriction is not necessarily true, then leave `symmetric_alpha=FALSE`.

Several other arguments may be provided as well:

- `estimate_pairwise_alpha_from_joint`: A boolean value indicating, when estimating reliability for pairs of cross-validation iterations, whether each pair of iterations should use the concentration parameters estimated across all iterations (`TRUE`) or whether separate sets of concentration parameters should be estimated for each pair of cross-validation iterations (`FALSE`). This parameter only has an effect when `alpha=NA`, and it is strongly suggested that the default value of `TRUE` be used for this parameter.

- `method`: A string indicating what approach (`"rotational"`) or (`"forward"`) will be used to reorder topics (see Britt, under review); the (`"forward"`) option will be implemented at a later date.

- `different_documents`: A boolean value indicating, if each list element in \code{x} represents the output from a [**topicmodels**::LDA] function call, whether some cross-validation iterations used different sets of documents than others.

- `pairwise`: A boolean value indicating whether pairwise reliability between individual cross-validation iterations should be computed. If `pairwise=FALSE`, pairwise reliability will be ignored in order to reduce the time and memory complexity of the computation.

- `samples`: An integer value indicating how many samples to use to estimate expected differences when `shuffle=TRUE`.

- `lower_bound`: A boolean value indicating whether the lower bound of the expected differences (based on i.i.d. beta distributions) should be used (`shuffle=TRUE`) rather than shuffling Dirichlet distributions to estimate the exact difference in every sample (`shuffle=FALSE`). For Dirichlet distributions with many categories and function calls with a large value of samples, setting `shuffle=TRUE` is strongly recommended, as reordering numerous samples with a large number of categories may require an unreasonable amount of time.

- `convcrit`: A numeric value indicating the threshold for convergence used to estimate concentration parameters when `alpha=NA`.

- `maxit`: A numeric value indicating the maximum number of iterations used to estimate concentration parameters when `alpha=NA`.

- `verbose`: A boolean value indicating whether `brittnu()` and its helper functions should provide progress updates.

- `zero`: A numeric value. If `type="wt"` and `different_documents=TRUE`, then whenever a word appears in some cross-validation iterations but not others, this value is assigned as the word's allocation to all topics within any cross-validation iteration in which the word did not originally appear.

- `zero2`: A numeric value. If `alpha=NA`, then in order to prevent errors when concentration parameters are being estimated, this is the minimum allocation permitted for any given category during the estimation procedure.

The output for the `brittnu` function is a list with the following four elements:

1. Britt's single-distribution nu for joint reliability
   - This is formatted as a vector whose length is equal to the number of Dirichlet distributions being evaluated (e.g., when computing word-topic reliability for LDA results, the vector's length is the number of topics).
   - Each element of the vector is Britt's single-distribution nu for the corresponding distribution, evaluated using all cross-validation iterations.
   - For instance, if your results are saved as an object called `reliability`, then `reliability[[1]]` is the vector representing Britt's joint single-distribution nu for all cross-validation iterations.
2. Britt's single-distribution nu for pairwise reliability
   - This is formatted as a list of lists, each of which contains a vector whose length is equal to the number of Dirichlet distributions being evaluated.
   - Each element of the vector is Britt's single-distribution nu for the corresponding distribution, evaluated using the specified pair of cross-validation iterations.
   - For instance, if your results are saved as an object called `reliability`, then `reliability[[2]][[5]][[7]]` is the vector representing Britt's pairwise single-distribution nu for the 5th and 7th cross-validation iterations.
     - Note that, to avoid recording redundant data, reversing the iterations (e.g., `reliability[[2]][[7]][[5]]`) will yield a `NULL` result.
3. Britt's omnibus (multiple-distribution) nu for joint reliability
   - This is formatted as a numeric value representing Britt's omnibus nu, evaluated using all cross-validation iterations.
   - For instance, if your results are saved as an object called `reliability`, then `reliability[[3]]` is the value representing Britt's joint omnibus nu for all cross-validation iterations.
4. Britt's omnibus (multiple-distribution) nu for pairwise reliability
   - This is formatted as a list of lists, each of which is a numeric value representing Britt's omnibus nu, evaluated for the speciusing all cross-validation iterations.
   - Each element of the vector is Britt's single-distribution nu for the corresponding distribution, evaluated using the specified pair of cross-validation iterations.
   - For instance, if your results are saved as an object called `reliability`, then `reliability[[4]][[5]][[7]]` is the vector representing Britt's pairwise omnibus nu for the 5th and 7th cross-validation iterations.
     - Note that, to avoid recording redundant data, reversing the iterations (e.g., `reliability[[4]][[7]][[5]]`) will yield a `NULL` result.



## Examples

```r
#Example 1: LDA 20-fold cross-validation using 40 topics
install.packages("devtools")
library(devtools)
install_github("bcbritt/brittnu")
library(brittnu)
install.packages("topicmodels") #Used in this example to generate data
library(topicmodels)
data("AssociatedPress")
set.seed(1797)
ap1 <- topicmodels::LDA(AssociatedPress, k = 40)
set.seed(1798)
ap2 <- topicmodels::LDA(AssociatedPress, k = 40)
set.seed(1799)
ap3 <- topicmodels::LDA(AssociatedPress, k = 40)
set.seed(1800)
ap4 <- topicmodels::LDA(AssociatedPress, k = 40)
set.seed(1801)
ap5 <- topicmodels::LDA(AssociatedPress, k = 40)
set.seed(1802)
ap6 <- topicmodels::LDA(AssociatedPress, k = 40)
set.seed(1803)
ap7 <- topicmodels::LDA(AssociatedPress, k = 40)
set.seed(1804)
ap8 <- topicmodels::LDA(AssociatedPress, k = 40)
set.seed(1805)
ap9 <- topicmodels::LDA(AssociatedPress, k = 40)
set.seed(1806)
ap10 <- topicmodels::LDA(AssociatedPress, k = 40)
set.seed(1807)
ap11 <- topicmodels::LDA(AssociatedPress, k = 40)
set.seed(1808)
ap12 <- topicmodels::LDA(AssociatedPress, k = 40)
set.seed(1809)
ap13 <- topicmodels::LDA(AssociatedPress, k = 40)
set.seed(1810)
ap14 <- topicmodels::LDA(AssociatedPress, k = 40)
set.seed(1811)
ap15 <- topicmodels::LDA(AssociatedPress, k = 40)
set.seed(1812)
ap16 <- topicmodels::LDA(AssociatedPress, k = 40)
set.seed(1813)
ap17 <- topicmodels::LDA(AssociatedPress, k = 40)
set.seed(1814)
ap18 <- topicmodels::LDA(AssociatedPress, k = 40)
set.seed(1815)
ap19 <- topicmodels::LDA(AssociatedPress, k = 40)
set.seed(1816)
ap20 <- topicmodels::LDA(AssociatedPress, k = 40)
ap_all_40 <- list(ap1,ap2,ap3,ap4,ap5,ap6,ap7,ap8,ap9,ap10,ap11,ap12,ap13,
                  ap14,ap15,ap16,ap17,ap18,ap19,ap20)
set.seed(1817)
#Assess topic-document reliability
reliability_40_td <- brittnu(ap_all_40, type="td", symmetric_alpha=TRUE,
                             shuffle=TRUE, samples=1000)
print(reliability_40_td[[1]]) #Britt's joint single-distribution nu
print(reliability_40_td[[2]]) #Britt's pairwise single-distribution nu
print(reliability_40_td[[3]]) #Britt's joint omnibus nu
print(reliability_40_td[[4]]) #Britt's pairwise omnibus nu
print(reliability_40_td) #All Britt's nu values for topic-document reliability
set.seed(1820)
#Assess word-topic reliability
reliability_40_wt <- brittnu(ap_all_40, type="wt", symmetric_alpha=TRUE,
                             shuffle=TRUE, samples=1000)
print(reliability_40_wt[[1]]) #Britt's joint single-distribution nu
print(reliability_40_wt[[2]]) #Britt's pairwise single-distribution nu
print(reliability_40_wt[[3]]) #Britt's joint omnibus nu
print(reliability_40_wt[[4]]) #Britt's pairwise omnibus nu
print(reliability_40_wt) #All Britt's nu values for word-topic reliability

#Example 2: Manually inputted data with known concentration parameters
install.packages("gtools") #Used in this example to generate data
library(gtools) #Used in this example to generate data
alpha1 <- c(1,1,1,1,1,1,1,1,1,1)
alpha2 <- c(2,2,2,2,2,2,2,2,2,2)
alpha3 <- c(0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1)
set.seed(2001)
data_with_known_alpha1 <- rbind(gtools::rdirichlet(1,alpha1),
                                gtools::rdirichlet(1,alpha2),
                                gtools::rdirichlet(1,alpha3))
set.seed(2002)
data_with_known_alpha2 <- rbind(gtools::rdirichlet(1,alpha1),
                                gtools::rdirichlet(1,alpha2),
                                gtools::rdirichlet(1,alpha3))
set.seed(2003)
data_with_known_alpha3 <- rbind(gtools::rdirichlet(1,alpha1),
                                gtools::rdirichlet(1,alpha2),
                                gtools::rdirichlet(1,alpha3))
set.seed(2004)
data_with_known_alpha4 <- rbind(gtools::rdirichlet(1,alpha1),
                                gtools::rdirichlet(1,alpha2),
                                gtools::rdirichlet(1,alpha3))
set.seed(2005)
data_with_known_alpha5 <- rbind(gtools::rdirichlet(1,alpha1),
                                gtools::rdirichlet(1,alpha2),
                                gtools::rdirichlet(1,alpha3))
data_with_known_alpha <- list(data_with_known_alpha1, data_with_known_alpha2,
                              data_with_known_alpha3, data_with_known_alpha4,
                              data_with_known_alpha5)
set.seed(2006)
brittnu(data_with_known_alpha,
        alpha=list(c(1,1,1,1,1,1,1,1,1,1), c(2,2,2,2,2,2,2,2,2,2),
                   c(0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1)))
```
