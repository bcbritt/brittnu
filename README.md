# brittnu: Britt's Nu and Euclidean Krippendorff's Alpha
This package computes two reliability coefficients. Britt's nu is used to assess reliability for data sets composed of multiple Dirichlet-distributed samples. Euclidean Krippendorff's alpha is used to assess reliability for Euclidean and compositional data that do not necessarily adhere to a Dirichlet distribution. These measures are especially useful for the cross-validation of topic models.



## Installation

To use this package, first install and load it in R with

```r
install.packages("devtools")
library(devtools)
install_github("bcbritt/brittnu")
library("brittnu")
```



## Purpose

Britt's nu and Euclidean Krippendorff's alpha are two families of reliability coefficients. Britt's nu is specifically tailored to Dirichlet-distributed data, such as the allocations of words to topics or of topics to documents in latent Dirichlet allocation (LDA). Euclidean Krippendorff's alpha, in turn, is more broadly applicable to any data that may be expressed as points in Euclidean space (regardless of the number of dimensions), such as compositional data, including that obtained via topic modeling techniques besides LDA.

Both Britt's nu and Euclidean Krippendorff's alpha allow the computation of up to four coefficients, encompassing the reliability for individual observations or the complete set of all observations, as well as the reliability for pairs of "raters" (e.g., cross-validation iterations) or the complete set of all raters.

- Pairwise reliability for a single observation*
- Pairwise reliability for all observations
- Omnibus (all raters) reliability for a single observation*
- Omnibus (all raters) reliability for all observations

*Euclidean Krippendorff's alpha may only be computed for a single observation, such as the allocations of words in a corpus to a single topic, using bootstrapping. Britt's nu may be computed for all four of the above combinations regardless of whether or not bootstrapping is used.

Each of these statistics is constructed using similar underlying logic as other interrater reliability coefficients.



## Usage

The `brittnu()` function generates all four forms of Britt's nu, and the `euclidkrip()` function generates all applicable forms of Euclidean Krippendorff's alpha (depending on whether bootstrapping is used).

`brittnu()` and `euclidkrip()` both require a list of vectors of Dirichlet-distributed observations (`x`). Each list element must be a 2D double vector with distributions as the rows and each category within each distribution as the columns (such that each row in the vector sums to 1).

Alternatively, for `brittnu()` in particular, each list element may instead be the result of a call to [**topicmodels**::LDA](https://www.rdocumentation.org/packages/topicmodels/versions/0.1-4/topics/LDA). For `brittnu()`, you should also specify `type="wt"` to assess word-topic reliability, `type="td"` for topic-document reliability, or `type=NA` if `x` consists of raw data rather than results from [**topicmodels**::LDA](https://www.rdocumentation.org/packages/topicmodels/versions/0.1-4/topics/LDA). Additionally, when using `brittnu()`, if the concentration parameters of the Dirichlet distribution are known, you should set `alpha` to a numeric vector of those concentration parameters. Otherwise, if `alpha=NA`, the concentration parameters will be estimated from the data. Likewise, many contexts, including LDA, often assume that the underlying distribution is symmetrical, such that all of the concentration parameters are identical. In such cases, you should set `symmetric_alpha=TRUE` in your call to `brittnu()`. If this restriction is not necessarily true, then leave `symmetric_alpha=FALSE`.

For either `brittnu()` or `euclidkrip()`, if the data emerged from a topic modeling approach (such as LDA), they may appear in a different order (or be "shuffled") for each cross-validation iteration, in which case they will need to be reordered before reliability is assessed. In such cases, you should set `shuffle=TRUE`. Otherwise, leave `shuffle=FALSE`. If `shuffle==TRUE`, then the `shuffle_method` argument may be used to specify how topics should be reordered, such as by using a topic-swapping procedure (`"rotational"`) or performing either an `"agglomerative"` or `"divisive"` hierarchical cluster analysis.

Several other arguments may be provided as well:

- `pairwise`: A boolean value indicating whether pairwise reliability between individual raters should be computed; if `FALSE`, pairwise reliability will be ignored in order to reduce the time and memory complexity of the computation

- `estimate_pairwise_alpha_from_joint` (`brittnu()` only): A boolean value indicating, when estimating reliability for pairs of raters, whether each pair of raters should use the concentration parameters estimated across all raters (`TRUE`) or whether separate sets of concentration parameters should be estimated for each pair of raters (`FALSE`); this argument only has an effect when `alpha==NA`, and it is strongly suggested that the default value of `TRUE` be used for this argument when possible

- `force_bootstrapping`: A boolean value indicating whether bootstrapping should be used to estimate expected differences rather than exactly computing them even if `shuffle==FALSE`

- `shuffle_dimension`: A string indicating whether `"rows"` or `"columns"` should be reordered; this argument only has an effect if `shuffle==TRUE` and, for `brittnu()` in particular, `type==NA`

- `clustering_method`: A string indicating the criterion that will be used to merge or divide clusters if `shuffle==TRUE` and either `shuffle_method=="agglomerative"` or `shuffle_method=="divisive"`; permitted values are `"average"`, `"minimum"`, `"maximum"`, `"median"`, `"centroid"`, and `"wald"`

- `slow_clustering`: A boolean value indicating whether the distances between clusters should be recalculated after each individual observation is added to the new cluster (`TRUE`), which can improve the cohesiveness of the resulting cluster, or whether all observations should be added to the new cluster without recomputing the distances between clusters after every addition (`FALSE`); this argument only has an effect if `shuffle==TRUE` and `shuffle_method=="divisive"`

- `samples`: An integer value indicating how many samples to use to estimate expected differences when `shuffle=TRUE`.

- `sampling_method` (`brittnu()` only; `euclidkrip()` uses `"nonparametric"`): A string indicating whether bootstrapped sampling should be performed using a `"parametric"` or `"nonparametric"` approach when `shuffle==TRUE`

- `lower_bound` A boolean value indicating whether the lower bound of the expected differences (based on i.i.d. beta distributions) should be used rather than reordering Dirichlet distributions to estimate the exact difference in every sample when `shuffle==TRUE`

- `convcrit` (`brittnu()` only): A numeric value indicating the threshold for convergence used to estimate concentration parameters when \code{alpha==NA}, `sampling_method=="parametric"`, and concentration parameters could not be directly obtained from the output of a [**topicmodels**::LDA](https://www.rdocumentation.org/packages/topicmodels/versions/0.1-4/topics/LDA) object provided as `x`

- `maxit` (`brittnu()` only): A numeric value indicating the maximum number of iterations used to estimate concentration parameters when `alpha==NA`, `sampling_method=="parametric"`, and concentration parameters could not be directly obtained from the output of a [**topicmodels**::LDA](https://www.rdocumentation.org/packages/topicmodels/versions/0.1-4/topics/LDA) object provided as `x`

- `progress` (`brittnu()` only): A boolean value indicating whether progress updates should be provided when estimating concentration parameters, if doing so is necessary

- `verbose`: A boolean value indicating whether `brittnu()`, `euclidkrip()`, and their helper functions should provide progress updates beyond the estimation of concentration parameters

- `different_documents` (`brittnu()` only): A boolean value indicating, if each list element in `x` represents the output from an [**topicmodels**::LDA](https://www.rdocumentation.org/packages/topicmodels/versions/0.1-4/topics/LDA) function call, whether some raters used different sets of documents than others

- `zero` (`brittnu()` only): A numeric value; if `type=="wt"` and `different_documents==TRUE`, then whenever a word appears in the portion of the corpus evaluated by some raters but not others, this value is assigned as its allocation to all topics for any rater that did not evaluate that word

- `tol` (`brittnu()` only): A numeric value representing the tolerance for observations whose sums are greater than or less than `1`; a warning message appears if the sum of the values for any observation is further from `1` than this value

As an additional note, if you are assessing the reliability of the allocations of words to topics via topic modeling, each cross-validation iteration may optionally use a different set of documents. In such cases, `different_documents` must be set to `TRUE`. For instance, if `mydata1`, `mydata2`, `mydata3`, `mydata4`, and `mydata5` each contain 80% of the documents from a single data set, then you could run

```r
{cv1 <- LDA(mydata1, k=15)
cv2 <- LDA(mydata2, k=15)
cv3 <- LDA(mydata3, k=15)
cv4 <- LDA(mydata4, k=15)
cv5 <- LDA(mydata5, k=15)
cv <- list(cv1, cv2, cv3, cv4, cv5)
rel <- brittnu(cv, type="wt", shuffle=TRUE, different_documents=TRUE)
summary(rel)}
```

to assess the word-topic allocation reliability for a 15-topic model. Any words that appear in some cross-validation iterations but not others will automatically be set to an allocation near 0 for any iterations in which they are absent from the data set. If you are instead assessing the reliability of the allocations of topics to documents, then if `different_documents==TRUE`, any iteration in which a given document does not appear will simply be excluded from the reliability computation for that document. Thus, fewer cross-validation iterations will be used to assess reliability for each individual document. This will weaken the stability of the reliability computation. Therefore, unless you are specifically assessing whether your topic model is stable regardless of what subset of documents was used to generate it, it is advised that all documents be included in all cross-validation iterations.

The output of `brittnu()` and `euclidkrip()` may be viewed using the `summary()` function. By default, this function only displays a single coefficient corresponding to all observations and all raters. Setting the `element` argument of this function will allow you to view the coefficients representing the pairwise reliability for a single observation (`"singlepairwise"`), the omnibus reliability for a single observation (`"singleomnibus"`), the pairwise reliability for all observations (`"multiplepairwise"`), the omnibus reliability for all observations (`"multipleomnibus"`), or all reliability coefficients at once (`"all"`).

If `shuffle==TRUE` for the call to `brittnu()` or `euclidkrip()`, then when calling `summary()` on the outputted object using `element=="singlepairwise"`, `element=="singleomnibus"`, or `element=="all"`, the matrix of reordered topics will be printed along with the reliability coefficients. Each row of this matrix indicates the manner in which the topics were reordered. For instance, if the first row is `c(3,1,2)`, that means that for the first rater, before computing reliability, the third topic from the original data set was moved to the first position, the first topic was moved to the second position, and the second topic was moved to the third position.



## Examples

```r
#Example 1: LDA, 3-fold cross-validation using 10 topics
install.packages("devtools")
library(devtools)
install_github("bcbritt/brittnu")
library(brittnu)
library(topicmodels)
data("AssociatedPress")
require(topicmodels)
data("AssociatedPress")
ap1 <- LDA(AssociatedPress, k = 10)
ap2 <- LDA(AssociatedPress, k = 10)
ap3 <- LDA(AssociatedPress, k = 10)
ap_all <- list(ap1,ap2,ap3)
#Topic-document reliability
reliability_ap_td <- brittnu(ap_all, type="td", symmetric_alpha=TRUE,
                             shuffle=TRUE, samples=1000, verbose=TRUE)
#Word-topic reliability
reliability_ap_wt <- brittnu(ap_all, type="wt", symmetric_alpha=TRUE,
                             shuffle=TRUE, samples=1000, verbose=TRUE)
summary(reliability_ap_td)
summary(reliability_ap_td, element="all")
summary(reliability_ap_wt)
summary(reliability_ap_wt, element="all")

#Example 2: Unshuffled Dirichlet data with known concentration parameters
require(gtools)
alpha1 <- c(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
alpha2 <- c(2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
alpha3 <- c(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
data_with_known_alpha1 <- rbind(gtools::rdirichlet(1,alpha1),
                                gtools::rdirichlet(1,alpha2),
                                gtools::rdirichlet(1,alpha3))
data_with_known_alpha2 <- rbind(gtools::rdirichlet(1,alpha1),
                                gtools::rdirichlet(1,alpha2),
                                gtools::rdirichlet(1,alpha3))
data_with_known_alpha3 <- rbind(gtools::rdirichlet(1,alpha1),
                                gtools::rdirichlet(1,alpha2),
                                gtools::rdirichlet(1,alpha3))
data_with_known_alpha4 <- rbind(gtools::rdirichlet(1,alpha1),
                                gtools::rdirichlet(1,alpha2),
                                gtools::rdirichlet(1,alpha3))
data_with_known_alpha5 <- rbind(gtools::rdirichlet(1,alpha1),
                                gtools::rdirichlet(1,alpha2),
                                gtools::rdirichlet(1,alpha3))
data_with_known_alpha <- list(data_with_known_alpha1, data_with_known_alpha2,
                              data_with_known_alpha3, data_with_known_alpha4,
                              data_with_known_alpha5)
reliability_known <- brittnu(data_with_known_alpha,
                             alpha=list(alpha1, alpha2, alpha3))
summary(reliability_known, element="all")

#Example 3: Compositional data, agglomerative clustering to reorder columns
rater1 <- rbind(c(0.80,0.05,0.15), c(0.10,0.85,0.05),
                c(0.05,0.05,0.90), c(0.85,0.10,0.05))
rater2 <- rbind(c(0.10,0.85,0.05), c(0.10,0.10,0.80),
                c(0.85,0.05,0.10), c(0.15,0.80,0.05))
data <- list(rater1, rater2)
reliability <- euclidkrip(data, shuffle=TRUE, shuffle_method="agglomerative",
                               shuffle_dimension="columns",
                               clustering_method="average", samples=500,
                               verbose=TRUE)
summary(reliability)
summary(reliability, element="all")
```