% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/brittnu.R
\name{brittnu}
\alias{brittnu}
\title{Britt's Nu}
\usage{
brittnu(
  x,
  type = NA,
  alpha = NA,
  symmetric_alpha = FALSE,
  pairwise = TRUE,
  estimate_pairwise_alpha_from_joint = TRUE,
  force_bootstrapping = FALSE,
  shuffle = FALSE,
  shuffle_method = "rotational",
  shuffle_dimension = NA,
  clustering_method = "average",
  slow_clustering = FALSE,
  samples = 1000,
  sampling_method = "parametric",
  lower_bound = FALSE,
  convcrit = 1e-05,
  maxit = 1000,
  progress = FALSE,
  verbose = FALSE,
  different_documents = FALSE,
  zero = (10^(-16)),
  tol = 1e-04
)
}
\arguments{
\item{x}{A list such that each element is either the output from a
\code{\link[topicmodels]{LDA}} function call or a 2D double vector with
observations as the rows and each category within each observation as the
columns (such that each row in the vector sums to 1)}

\item{type}{A string indicating the type of reliability being assessed, with
permitted values of \code{"wt"} (word-topic reliability), \code{"td"}
(topic-document reliability), and NA; if \code{x} contains the output from
multiple \code{\link[topicmodels]{LDA}} function calls, this must be set to
either \code{"wt"} (word-topic reliability) or \code{"td"} (topic-document
reliability)}

\item{alpha}{A list whose length is equal to the number of observations, with
each list element representing a vector comprising the concentration
parameters for the categories in the corresponding observation; if this is
\code{NA} and \code{sampling_method=="parametric"}, the concentration
parameters will be estimated from \code{x}}

\item{symmetric_alpha}{A boolean value indicating whether concentration
parameters should be assumed to be unchanged between observations, which is
common in many topic modeling procedures; this parameter only has an effect
when \code{alpha=NA}}

\item{pairwise}{A boolean value indicating whether pairwise reliability
between individual raters should be computed; if \code{FALSE}, pairwise
reliability will be ignored in order to reduce the time and memory
complexity of the computation}

\item{estimate_pairwise_alpha_from_joint}{A boolean value indicating, when
estimating reliability for pairs of raters, whether each pair of raters
should use the concentration parameters estimated across all raters
(\code{TRUE}) or whether separate sets of concentration parameters should
be estimated for each pair of raters (\code{FALSE}); this argument only has
an effect when \code{alpha==NA}, and it is strongly suggested that the
default value of \code{TRUE} be used for this argument when possible}

\item{force_bootstrapping}{A boolean value indicating whether bootstrapping
should be used to estimate expected differences rather than exactly
computing them even if \code{shuffle==FALSE}}

\item{shuffle}{A boolean value indicating whether or not the topics may have
been shuffled into different sequences by each rater; if this is
\code{TRUE} and \code{shuffle_dimension="rows"}, then the observations
(rows) corresponding to each rater will be reordered to achieve a (local)
optimal fit, whereas if this is \code{TRUE} and
\code{shuffle_dimension="columns"}, the categories (columns) will be
reordered instead}

\item{shuffle_method}{A string indicating what method (\code{"rotational"},
\code{"agglomerative"}, or \code{"divisive"}) will be used to reorder
topics}

\item{shuffle_dimension}{A string indicating whether \code{"rows"} or
\code{"columns"} should be reordered; this argument only has an effect if
\code{shuffle==TRUE} and \code{type==NA}}

\item{clustering_method}{A string indicating the criterion that will be used
to merge or divide clusters if \code{shuffle==TRUE} and either
\code{shuffle_method=="agglomerative"} or
\code{shuffle_method=="divisive"}; permitted values are \code{"average"},
\code{"minimum"}, \code{"maximum"}, \code{"median"}, \code{"centroid"}, and
\code{"wald"}}

\item{slow_clustering}{A boolean value indicating whether the distances
between clusters should be recalculated after each individual observation
is added to the new cluster (\code{TRUE}), which can improve the
cohesiveness of the resulting cluster, or whether all observations should
be added to the new cluster without recomputing the distances between
clusters after every addition (\code{FALSE}); this argument only has an
effect if \code{shuffle==TRUE} and \code{shuffle_method=="divisive"}}

\item{samples}{An integer value indicating how many samples to use to
estimate expected differences when \code{shuffle==TRUE}}

\item{sampling_method}{A string indicating whether bootstrapped sampling
should be performed using a \code{"parametric"} or \code{"nonparametric"}
approach when \code{shuffle==TRUE}}

\item{lower_bound}{A boolean value indicating whether the lower bound of the
expected differences (based on i.i.d. beta distributions) should be used
rather than shuffling Dirichlet distributions to estimate the exact
difference in every sample when \code{shuffle==TRUE}}

\item{convcrit}{A numeric value indicating the threshold for convergence used
to estimate concentration parameters when \code{alpha==NA},
\code{sampling_method=="parametric"}, and concentration parameters could
not be directly obtained from the output of a
\code{\link[topicmodels]{LDA}} object provided as \code{x}}

\item{maxit}{A numeric value indicating the maximum number of iterations used
to estimate concentration parameters when \code{alpha==NA},
\code{sampling_method=="parametric"}, and concentration parameters could
not be directly obtained from the output of a
\code{\link[topicmodels]{LDA}} object provided as \code{x}}

\item{progress}{A boolean value indicating whether progress updates should be
provided when estimating concentration parameters, if doing so is necessary}

\item{verbose}{A boolean value indicating whether \link{brittnu} and its
helper functions should provide progress updates beyond the estimation of
concentration parameters}

\item{different_documents}{A boolean value indicating, if each list element
in \code{x} represents the output from an \code{\link[topicmodels]{LDA}}
function call, whether some raters used different sets of documents than
others}

\item{zero}{A numeric value; if \code{type=="wt"} and
\code{different_documents==TRUE}, then whenever a word appears in the
portion of the corpus evaluated by some raters but not others, this value
is assigned as its allocation to all topics for any rater that did not
evaluate that word}

\item{tol}{A numeric value representing the tolerance for observations whose
sums are greater than or less than \code{1}; a warning message appears if
the sum of the values for any observation is further from \code{1} than
this value}
}
\value{
A list of class \code{brittnu_rel} containing seven elements: Britt's
  single-observation nu for all raters (as a numeric vector indicating the
  reliability for each observation), Britt's single-observation nu for pairs
  of raters (as a list containing lists containing numeric vectors indicating
  the reliability of each observation for each pair of raters, e.g., the
  second element of the first list contains the reliability for raters 1 and
  2), Britt's multiple-observation nu for all raters (as a numeric value),
  Britt's multiple-observation nu for pairs of raters (as a list containing
  lists containing numeric values, e.g., the second element of the first list
  contains the reliability for raters 1 and 2), the matrix of reordered
  topics for each rater (if \code{shuffle==TRUE}), any warnings raised, and
  the value of the \code{type} argument
}
\description{
This function computes the Britt's nu reliability coefficient
for data sets composed of multiple Dirichlet-distributed deviates, as
described by Britt (under review).
}
\details{
One of the most important uses for Britt's nu is to assess the
reliability of the allocations of words to topics or of topics to documents
via topic modeling techniques such as latent Dirichlet allocation (LDA). When
the results of multiple LDA cross-validation iterations obtained via
\code{\link[topicmodels]{LDA}} are provided as \code{x}, the \code{type}
parameter should be set to either \code{"wt"} or \code{"td"} to indicate
whether word-topic or topic-document reliability, respectively, should be
assessed.

Crucially, different topic modeling cross-validation iterations (whether LDA
or otherwise), which effectively represent distinct raters of the same data
set, may result in the same topics appearing in a different sequence. As
such, those topics, which may represent either the rows or the columns of the
data set, often need to be reordered in order to yield an optimal fit.

In practice, it is generally infeasible to assess every possible combination
of topics across all raters. As such, \link{brittnu} provides three methods
of converging toward an optimal topic order for each rater. The default,
\code{shuffle_method="rotational"}, takes the provided sequence and swaps
pairs of categories until no further swaps would further improve the fit.
\code{shuffle_method="agglomerative"} and \code{shuffle_method="divisive"}
instead perform hierarchical cluster analyses of all topics, with the added
restriction that two topics constructed by the same rater may not appear in
the same cluster. Notably, the \code{"agglomerative"} option can take
excessive time in some cases, while the \code{"divisive"} option forms
clusters based on the macro-level dynamics of the data and may therefore
yield weakly matched sets of individual topics compared to the other
available options.

Regardless of whether the rows or columns of the data set must be reordered,
when estimating Britt's nu, it is common to compute four separate
coefficients in this family: Britt's single-observation nu for all
raters, Britt's single-observation nu for pairs of raters, Britt's omnibus nu
for all raters, and Britt's single-observation nu for pairs of raters. For
example, when assessing the reliability of multiple latent Dirichlet
allocation cross-validation folds that each used the same data set and number
of topics, single-observation reliability can be used to assess the
reliability of the allocations of words to each individual topic, while
multiple-observation reliability indicates the reliability of the allocations
of words to the set of all topics. Likewise, the coefficients that take all
raters into account generally more valuable for the summative assessment of
reliability, but the pairwise coefficients are sometimes useful for
diagnostic purposes.

By default, \link{brittnu} provides all four reliability coefficients. To
reduce the time and memory required, the pairwise versions may optionally be
omitted from the computation by specifying \code{pairwise=FALSE}.

Also by default, the expected difference component of Britt's nu is estimated
using parametric bootstrapping based on the distribution of the original
data. This process can sometimes become computationally intensive. In such
cases, it may be advisable to set \code{sampling_method="nonparametric"} in
order to use nonparametric bootstrapping rather than parametric
bootstrapping. You may also consider setting \code{lower_bound=TRUE} to use
i.i.d. beta distributions to estimate the lower bound of the expected
differences between observations, ultimately yielding an estimated lower
bound for Britt's nu itself. This method may be faster than some methods of
reordering rows or columns in some cases, although this is not always true.

Additional usage notes:

1. When \code{x} is a list of 2D double vectors (rather than a list of
objects outputted from \code{\link[topicmodels]{LDA}}), each row of each list
element should be a single observation from a Dirichlet distribution (e.g.,
an LDA topic), and each column should be a category within that distribution
(e.g., a word).

2. It is generally recommended that \code{estimate_pairwise_alpha_from_joint}
be set to \code{TRUE}, which is the default setting. If concentration
parameters must be estimated from \code{x}, and if all elements were
generated via LDA from the same corpus or are otherwise assumed to have
emerged from the same underlying distribution, then there is little reason to
expect different sets of concentration parameters to be necessary across
iterations. You should only consider setting
\code{estimate_pairwise_alpha_from_joint=FALSE} if different data sets that
may not have come from the same underlying distribution were used to generate
different elements of \code{x}. Doing so, however, may substantially slow the
procedure, so it is generally not recommended unless essential.

3. If you are assessing the reliability of the allocations of words to topics
different set of documents. In such cases, \code{different_documents} must be
set to \code{TRUE}. For instance, if \code{mydata1}, \code{mydata2},
\code{mydata3}, \code{mydata4}, and \code{mydata5} each contain 80% of the
documents from a single data set, then you could run

\preformatted{cv1 <- LDA(mydata1, k=15)
cv2 <- LDA(mydata2, k=15)
cv3 <- LDA(mydata3, k=15)
cv4 <- LDA(mydata4, k=15)
cv5 <- LDA(mydata5, k=15)
cv <- list(cv1, cv2, cv3, cv4, cv5)
rel <- brittnu(cv, type="wt", shuffle=TRUE, different_documents=TRUE)
summary(rel)}

to assess the word-topic allocation reliability for a 15-topic model. Any
words that appear in some cross-validation iterations but not others will
automatically be set to an allocation near 0 for any iterations in which they
are absent from the data set. If you are instead assessing the reliability of
the allocations of topics to documents, then if
\code{different_documents==TRUE}, any iteration in which a given document
does not appear will simply be excluded from the reliability computation for
that document. Thus, fewer cross-validation iterations will be used to assess
reliability for each individual document. This will weaken the stability of
the reliability computation. Therefore, unless you are specifically assessing
whether your topic model is stable regardless of what subset of documents was
used to generate it, it is advised that all documents be included in all
cross-validation iterations.

4. When \code{shuffle==TRUE}, each element in \code{x} comprises a large
number of topics, and/or \code{samples} is large, setting
\code{lower_bound=TRUE} may sometimes be useful in order to eliminate the
need to reorder a large number of topics in numerous bootstrapped samples.
Additionally, for Dirichlet distributions with many categories, precise
estimates of the concentration parameters may require excessive time, so
whenever possible, a priori known concentration parameters should be
provided via the \code{alpha} parameter rather than estimating them from the
data. This is also useful to ensure the validity of Britt's nu. When this is
not possible, consider setting \code{sampling_method="nonparametric"} in
order to avoid the use of concentration parameters altogether. Alternatively,
the procedure may be expedited by changing \code{convcrit} from its default
value (0.00001) to a larger number. You may also wish to set
\code{progress=TRUE} and \code{verbose=TRUE} to receive periodic progress
updates and ensure that the computation is proceeding as expected.

5. When \code{shuffle==TRUE}, the topics in \code{x} are reordered. This is
described in the matrix of reordered topics, which is provided as one of the
elements of the list returned by \link{brittnu} and can be viewed using
\code{summary()} on that object. Each row of this matrix indicates the manner
in which the topics were reordered. For instance, if the first row is
c(3,1,2), that means that for the first rater, the third topic was moved to
the first position, the first topic was moved to the second position, and the
second topic was moved to the third position.
}
\section{References}{

  Britt, B. C. (under review). Interrater reliability for compositional,
  Euclidean, and Dirichlet Data.
}

\examples{
#Example 1: LDA results
require(topicmodels)
data("AssociatedPress")
ap1 <-  LDA(AssociatedPress, k = 10)
ap2 <-  LDA(AssociatedPress, k = 10)
ap3 <-  LDA(AssociatedPress, k = 10)
ap_all <- list(ap1,ap2,ap3)
reliability_ap_td <- brittnu(ap_all, type="td", symmetric_alpha=TRUE,
                             shuffle=TRUE, samples=1000, verbose=TRUE)
reliability_ap_wt <- brittnu(ap_all, type="wt", symmetric_alpha=TRUE,
                             shuffle=TRUE, samples=1000, verbose=TRUE)
summary(reliability_ap_td)
summary(reliability_ap_td, element="all")
summary(reliability_ap_wt)
summary(reliability_ap_wt, element="all")

#Example 2: Manually inputted data with known concentration parameters
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
}
