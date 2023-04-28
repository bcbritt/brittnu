#' Britt's Nu
#'
#' @description This function computes the Britt's nu reliability coefficient
#' for data sets composed of multiple Dirichlet-distributed deviates, as
#' described by Britt (under review).
#'
#' @details One of the most important uses for Britt's nu is to assess the
#' reliability of the allocations of words to topics or of topics to documents
#' via topic modeling techniques such as latent Dirichlet allocation (LDA). When
#' the results of multiple LDA cross-validation iterations obtained via
#' \code{\link[topicmodels]{LDA}} are provided as \code{x}, the \code{type}
#' parameter should be set to either \code{"wt"} or \code{"td"} to indicate
#' whether word-topic or topic-document reliability, respectively, should be
#' assessed.
#'
#' Crucially, different topic modeling cross-validation iterations (whether LDA
#' or otherwise), which effectively represent distinct raters of the same data
#' set, may result in the same topics appearing in a different sequence. As
#' such, those topics, which may represent either the rows or the columns of the
#' data set, often need to be reordered in order to yield an optimal fit.
#'
#' In practice, it is generally infeasible to assess every possible combination
#' of topics across all raters. As such, \link{brittnu} provides three methods
#' of converging toward an optimal topic order for each rater. The default,
#' \code{shuffle_method="rotational"}, takes the provided sequence and swaps
#' pairs of categories until no further swaps would further improve the fit.
#' \code{shuffle_method="agglomerative"} and \code{shuffle_method="divisive"}
#' instead perform hierarchical cluster analyses of all topics, with the added
#' restriction that two topics constructed by the same rater may not appear in
#' the same cluster. Notably, the \code{"agglomerative"} option can take
#' excessive time in some cases, while the \code{"divisive"} option forms
#' clusters based on the macro-level dynamics of the data and may therefore
#' yield weakly matched sets of individual topics compared to the other
#' available options.
#'
#' Regardless of whether the rows or columns of the data set must be reordered,
#' when estimating Britt's nu, it is common to compute four separate
#' coefficients in this family: Britt's single-observation nu for all
#' raters, Britt's single-observation nu for pairs of raters, Britt's omnibus nu
#' for all raters, and Britt's single-observation nu for pairs of raters. For
#' example, when assessing the reliability of multiple latent Dirichlet
#' allocation cross-validation folds that each used the same data set and number
#' of topics, single-observation reliability can be used to assess the
#' reliability of the allocations of words to each individual topic, while
#' multiple-observation reliability indicates the reliability of the allocations
#' of words to the set of all topics. Likewise, the coefficients that take all
#' raters into account generally more valuable for the summative assessment of
#' reliability, but the pairwise coefficients are sometimes useful for
#' diagnostic purposes.
#'
#' By default, \link{brittnu} provides all four reliability coefficients. To
#' reduce the time and memory required, the pairwise versions may optionally be
#' omitted from the computation by specifying \code{pairwise=FALSE}.
#'
#' Also by default, the expected difference component of Britt's nu is estimated
#' using parametric bootstrapping based on the distribution of the original
#' data. This process can sometimes become computationally intensive. In such
#' cases, it may be advisable to set \code{sampling_method="nonparametric"} in
#' order to use nonparametric bootstrapping rather than parametric
#' bootstrapping. You may also consider setting \code{lower_bound=TRUE} to use
#' i.i.d. beta distributions to estimate the lower bound of the expected
#' differences between observations, ultimately yielding an estimated lower
#' bound for Britt's nu itself. This method may be faster than some methods of
#' reordering rows or columns in some cases, although this is not always true.
#'
#' Additional usage notes:
#'
#' 1. When \code{x} is a list of 2D double vectors (rather than a list of
#' objects outputted from \code{\link[topicmodels]{LDA}}), each row of each list
#' element should be a single observation from a Dirichlet distribution (e.g.,
#' an LDA topic), and each column should be a category within that distribution
#' (e.g., a word).
#'
#' 2. It is generally recommended that \code{estimate_pairwise_alpha_from_joint}
#' be set to \code{TRUE}, which is the default setting. If concentration
#' parameters must be estimated from \code{x}, and if all elements were
#' generated via LDA from the same corpus or are otherwise assumed to have
#' emerged from the same underlying distribution, then there is little reason to
#' expect different sets of concentration parameters to be necessary across
#' iterations. You should only consider setting
#' \code{estimate_pairwise_alpha_from_joint=FALSE} if different data sets that
#' may not have come from the same underlying distribution were used to generate
#' different elements of \code{x}. Doing so, however, may substantially slow the
#' procedure, so it is generally not recommended unless essential.
#'
#' 3. If you are assessing the reliability of the allocations of words to topics
#  via topic modeling, each cross-validation iteration may optionally use a
#' different set of documents. In such cases, \code{different_documents} must be
#' set to \code{TRUE}. For instance, if \code{mydata1}, \code{mydata2},
#' \code{mydata3}, \code{mydata4}, and \code{mydata5} each contain 80% of the
#' documents from a single data set, then you could run
#'
#' \preformatted{cv1 <- LDA(mydata1, k=15)
#' cv2 <- LDA(mydata2, k=15)
#' cv3 <- LDA(mydata3, k=15)
#' cv4 <- LDA(mydata4, k=15)
#' cv5 <- LDA(mydata5, k=15)
#' cv <- list(cv1, cv2, cv3, cv4, cv5)
#' rel <- brittnu(cv, type="wt", shuffle=TRUE, different_documents=TRUE)
#' summary(rel)}
#'
#' to assess the word-topic allocation reliability for a 15-topic model. Any
#' words that appear in some cross-validation iterations but not others will
#' automatically be set to an allocation near 0 for any iterations in which they
#' are absent from the data set. If you are instead assessing the reliability of
#' the allocations of topics to documents, then if
#' \code{different_documents==TRUE}, any iteration in which a given document
#' does not appear will simply be excluded from the reliability computation for
#' that document. Thus, fewer cross-validation iterations will be used to assess
#' reliability for each individual document. This will weaken the stability of
#' the reliability computation. Therefore, unless you are specifically assessing
#' whether your topic model is stable regardless of what subset of documents was
#' used to generate it, it is advised that all documents be included in all
#' cross-validation iterations.
#'
#' 4. When \code{shuffle==TRUE}, each element in \code{x} comprises a large
#' number of topics, and/or \code{samples} is large, setting
#' \code{lower_bound=TRUE} may sometimes be useful in order to eliminate the
#' need to reorder a large number of topics in numerous bootstrapped samples.
#' Additionally, for Dirichlet distributions with many categories, precise
#' estimates of the concentration parameters may require excessive time, so
#' whenever possible, a priori known concentration parameters should be
#' provided via the \code{alpha} parameter rather than estimating them from the
#' data. This is also useful to ensure the validity of Britt's nu. When this is
#' not possible, consider setting \code{sampling_method="nonparametric"} in
#' order to avoid the use of concentration parameters altogether. Alternatively,
#' the procedure may be expedited by changing \code{convcrit} from its default
#' value (0.00001) to a larger number. You may also wish to set
#' \code{progress=TRUE} and \code{verbose=TRUE} to receive periodic progress
#' updates and ensure that the computation is proceeding as expected.
#'
#' 5. When \code{shuffle==TRUE}, the topics in \code{x} are reordered. This is
#' described in the matrix of reordered topics, which is provided as one of the
#' elements of the list returned by \link{brittnu} and can be viewed using
#' \code{summary()} on that object. Each row of this matrix indicates the manner
#' in which the topics were reordered. For instance, if the first row is
#' c(3,1,2), that means that for the first rater, the third topic was moved to
#' the first position, the first topic was moved to the second position, and the
#' second topic was moved to the third position.
#'
#' @param x A list such that each element is either the output from a
#'   \code{\link[topicmodels]{LDA}} function call or a 2D double vector with
#'   observations as the rows and each category within each observation as the
#'   columns (such that each row in the vector sums to 1)
#' @param type A string indicating the type of reliability being assessed, with
#'   permitted values of \code{"wt"} (word-topic reliability), \code{"td"}
#'   (topic-document reliability), and NA; if \code{x} contains the output from
#'   multiple \code{\link[topicmodels]{LDA}} function calls, this must be set to
#'   either \code{"wt"} (word-topic reliability) or \code{"td"} (topic-document
#'   reliability)
#' @param alpha A list whose length is equal to the number of observations, with
#'   each list element representing a vector comprising the concentration
#'   parameters for the categories in the corresponding observation; if this is
#'   \code{NA} and \code{sampling_method=="parametric"}, the concentration
#'   parameters will be estimated from \code{x}
#' @param symmetric_alpha A boolean value indicating whether concentration
#'   parameters should be assumed to be unchanged between observations, which is
#'   common in many topic modeling procedures; this parameter only has an effect
#'   when \code{alpha=NA}
#' @param pairwise A boolean value indicating whether pairwise reliability
#'   between individual raters should be computed; if \code{FALSE}, pairwise
#'   reliability will be ignored in order to reduce the time and memory
#'   complexity of the computation
#' @param estimate_pairwise_alpha_from_joint A boolean value indicating, when
#'   estimating reliability for pairs of raters, whether each pair of raters
#'   should use the concentration parameters estimated across all raters
#'   (\code{TRUE}) or whether separate sets of concentration parameters should
#'   be estimated for each pair of raters (\code{FALSE}); this argument only has
#'   an effect when \code{alpha==NA}, and it is strongly suggested that the
#'   default value of \code{TRUE} be used for this argument when possible
#' @param force_bootstrapping A boolean value indicating whether bootstrapping
#'   should be used to estimate expected differences rather than exactly
#'   computing them even if \code{shuffle==FALSE}
#' @param shuffle A boolean value indicating whether or not the topics may have
#'   been shuffled into different sequences by each rater; if this is
#'   \code{TRUE} and \code{shuffle_dimension="rows"}, then the observations
#'   (rows) corresponding to each rater will be reordered to achieve a (local)
#'   optimal fit, whereas if this is \code{TRUE} and
#'   \code{shuffle_dimension="columns"}, the categories (columns) will be
#'   reordered instead
#' @param shuffle_method A string indicating what method (\code{"rotational"},
#'   \code{"agglomerative"}, or \code{"divisive"}) will be used to reorder
#'   topics
#' @param shuffle_dimension A string indicating whether \code{"rows"} or
#'   \code{"columns"} should be reordered; this argument only has an effect if
#'   \code{shuffle==TRUE} and \code{type==NA}
#' @param clustering_method A string indicating the criterion that will be used
#'   to merge or divide clusters if \code{shuffle==TRUE} and either
#'   \code{shuffle_method=="agglomerative"} or
#'   \code{shuffle_method=="divisive"}; permitted values are \code{"average"},
#'   \code{"minimum"}, \code{"maximum"}, \code{"median"}, \code{"centroid"}, and
#'   \code{"wald"}
#' @param slow_clustering A boolean value indicating whether the distances
#'   between clusters should be recalculated after each individual observation
#'   is added to the new cluster (\code{TRUE}), which can improve the
#'   cohesiveness of the resulting cluster, or whether all observations should
#'   be added to the new cluster without recomputing the distances between
#'   clusters after every addition (\code{FALSE}); this argument only has an
#'   effect if \code{shuffle==TRUE} and \code{shuffle_method=="divisive"}
#' @param samples An integer value indicating how many samples to use to
#'   estimate expected differences when \code{shuffle==TRUE}
#' @param sampling_method A string indicating whether bootstrapped sampling
#'   should be performed using a \code{"parametric"} or \code{"nonparametric"}
#'   approach when \code{shuffle==TRUE}
#' @param lower_bound A boolean value indicating whether the lower bound of the
#'   expected differences (based on i.i.d. beta distributions) should be used
#'   rather than shuffling Dirichlet distributions to estimate the exact
#'   difference in every sample when \code{shuffle==TRUE}
#' @param convcrit A numeric value indicating the threshold for convergence used
#'   to estimate concentration parameters when \code{alpha==NA},
#'   \code{sampling_method=="parametric"}, and concentration parameters could
#'   not be directly obtained from the output of a
#'   \code{\link[topicmodels]{LDA}} object provided as \code{x}
#' @param maxit A numeric value indicating the maximum number of iterations used
#'   to estimate concentration parameters when \code{alpha==NA},
#'   \code{sampling_method=="parametric"}, and concentration parameters could
#'   not be directly obtained from the output of a
#'   \code{\link[topicmodels]{LDA}} object provided as \code{x}
#' @param progress A boolean value indicating whether progress updates should be
#'   provided when estimating concentration parameters, if doing so is necessary
#' @param verbose A boolean value indicating whether \link{brittnu} and its
#'   helper functions should provide progress updates beyond the estimation of
#'   concentration parameters
#' @param different_documents A boolean value indicating, if each list element
#'   in \code{x} represents the output from an \code{\link[topicmodels]{LDA}}
#'   function call, whether some raters used different sets of documents than
#'   others
#' @param zero A numeric value; if \code{type=="wt"} and
#'   \code{different_documents==TRUE}, then whenever a word appears in the
#'   portion of the corpus evaluated by some raters but not others, this value
#'   is assigned as its allocation to all topics for any rater that did not
#'   evaluate that word
#' @param tol A numeric value representing the tolerance for observations whose
#'   sums are greater than or less than \code{1}; a warning message appears if
#'   the sum of the values for any observation is further from \code{1} than
#'   this value
#' @return A list of class \code{brittnu_rel} containing seven elements: Britt's
#'   single-observation nu for all raters (as a numeric vector indicating the
#'   reliability for each observation), Britt's single-observation nu for pairs
#'   of raters (as a list containing lists containing numeric vectors indicating
#'   the reliability of each observation for each pair of raters, e.g., the
#'   second element of the first list contains the reliability for raters 1 and
#'   2), Britt's multiple-observation nu for all raters (as a numeric value),
#'   Britt's multiple-observation nu for pairs of raters (as a list containing
#'   lists containing numeric values, e.g., the second element of the first list
#'   contains the reliability for raters 1 and 2), the matrix of reordered
#'   topics for each rater (if \code{shuffle==TRUE}), any warnings raised, and
#'   the value of the \code{type} argument
#' @importFrom "utils" "flush.console"
#' @importFrom "stats" "rgamma"
#' @importFrom "methods" "slot"
#' @importFrom "plyr" "rbind.fill.matrix"
#' @importFrom "topicmodels" "LDA"
#' @section References:
#'   Britt, B. C. (under review). Interrater reliability for compositional,
#'   Euclidean, and Dirichlet Data.
#' @examples
#' #Example 1: LDA results
#' require(topicmodels)
#' data("AssociatedPress")
#' ap1 <-  LDA(AssociatedPress, k = 10)
#' ap2 <-  LDA(AssociatedPress, k = 10)
#' ap3 <-  LDA(AssociatedPress, k = 10)
#' ap_all <- list(ap1,ap2,ap3)
#' reliability_ap_td <- brittnu(ap_all, type="td", symmetric_alpha=TRUE,
#'                              shuffle=TRUE, samples=1000, verbose=TRUE)
#' reliability_ap_wt <- brittnu(ap_all, type="wt", symmetric_alpha=TRUE,
#'                              shuffle=TRUE, samples=1000, verbose=TRUE)
#' summary(reliability_ap_td)
#' summary(reliability_ap_td, element="all")
#' summary(reliability_ap_wt)
#' summary(reliability_ap_wt, element="all")
#'
#' #Example 2: Manually inputted data with known concentration parameters
#' require(gtools)
#' alpha1 <- c(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
#' alpha2 <- c(2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
#' alpha3 <- c(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
#' data_with_known_alpha1 <- rbind(gtools::rdirichlet(1,alpha1),
#'                                 gtools::rdirichlet(1,alpha2),
#'                                 gtools::rdirichlet(1,alpha3))
#' data_with_known_alpha2 <- rbind(gtools::rdirichlet(1,alpha1),
#'                                 gtools::rdirichlet(1,alpha2),
#'                                 gtools::rdirichlet(1,alpha3))
#' data_with_known_alpha3 <- rbind(gtools::rdirichlet(1,alpha1),
#'                                 gtools::rdirichlet(1,alpha2),
#'                                 gtools::rdirichlet(1,alpha3))
#' data_with_known_alpha4 <- rbind(gtools::rdirichlet(1,alpha1),
#'                                 gtools::rdirichlet(1,alpha2),
#'                                 gtools::rdirichlet(1,alpha3))
#' data_with_known_alpha5 <- rbind(gtools::rdirichlet(1,alpha1),
#'                                 gtools::rdirichlet(1,alpha2),
#'                                 gtools::rdirichlet(1,alpha3))
#' data_with_known_alpha <- list(data_with_known_alpha1, data_with_known_alpha2,
#'                               data_with_known_alpha3, data_with_known_alpha4,
#'                               data_with_known_alpha5)
#' reliability_known <- brittnu(data_with_known_alpha,
#'                              alpha=list(alpha1, alpha2, alpha3))
#' summary(reliability_known, element="all")
#' @export
#' brittnu

brittnu <- function(x, type=NA, alpha=NA, symmetric_alpha=FALSE, pairwise=TRUE,
                    estimate_pairwise_alpha_from_joint=TRUE, force_bootstrapping=FALSE,
                    shuffle=FALSE, shuffle_method="rotational", shuffle_dimension=NA,
                    clustering_method="average", slow_clustering=FALSE, samples=1000,
                    sampling_method="parametric", lower_bound=FALSE,
                    convcrit=0.00001, maxit=1000, progress=FALSE, verbose=FALSE,
                    different_documents=FALSE, zero=(10^(-16)), tol=0.0001) {

   warnings <- vector("list", 5) #One vector for each of the individual reliability types, and a separate (fifth) vector for when all reliabilities are printed
   for(i in 1:length(warnings)) {
      warnings[[i]] <- vector("character")
   }

   if(!is.na(type)) {
      if(type=="tw") {
         type="wt"
      }
      if(type=="dt") {
         type="td"
      }
   }

   for(i in 1:length(warnings)) {
      warnings[[i]] <- vector("character")
   }

   if(different_documents) {
      warning("The different_documents argument is currently experimental. It is strongly\nrecommended that you set different_documents=FALSE.")
      for(i in 1:length(warnings)) {
         warnings[[i]][length(warnings[[i]])+1] <- "The different_documents argument is currently experimental. It is strongly\n   recommended that you set different_documents=FALSE."
      }
   }

   if(shuffle==FALSE & lower_bound==TRUE) {
      warning("When shuffle=FALSE, there is no need to compute the lower bound for Britt's nu\nthat could occur based on category or observation reordering, so\nlower_bound=FALSE was ignored.")
      for(i in 1:length(warnings)) {
         warnings[[i]][length(warnings[[i]])+1] <- "When shuffle=FALSE, there is no need to compute the lower bound for Britt's\n   nu that could occur based on category or observation reordering, so\n   lower_bound=FALSE was ignored."
      }
      lower_bound = FALSE
   }

   if(is.na(type)) {
      shuffle_dimension = "rows"
   } else {
      if(type=="td") {
         if(!is.na(shuffle_dimension)) {
            if(shuffle_dimension == "rows") {
               warning("When type='td', each column represents a distinct latent topic. If the same\ntopic was placed in a different column by different 'raters' (LDA iterations),\nthen those columns will need to be reordered, not the rows. Therefore, the\nshuffle_dimension='rows' argument was ignored.")
               for(i in 1:length(warnings)) {
                  warnings[[i]][length(warnings[[i]])+1] <- "When type='td', each column represents a distinct latent topic. If the same\n   topic was placed in a different column by different 'raters' (LDA\n   iterations), then those columns will need to be reordered, not the rows.\n   Therefore, the shuffle_dimension='rows' argument was ignored."
               }
            }
         }
         shuffle_dimension = "columns"
      }

      if(type=="wt") {
         if(!is.na(shuffle_dimension)) {
            if(shuffle_dimension == "columns") {
               warning("When type='wt', each row represents a distinct latent topic. If the same topic\nwas placed in a different row by different 'raters' (LDA iterations), then those\nrows will need to be reordered, not the columns. Therefore, the\nshuffle_dimension='columns' argument was ignored.")
               for(i in 1:length(warnings)) {
                  warnings[[i]][length(warnings[[i]])+1] <- "When type='wt', each row represents a distinct latent topic. If the same\n   topic was placed in a different row by different 'raters' (LDA iterations),\n   then those rows will need to be reordered, not the columns. Therefore, the\n   shuffle_dimension='columns' argument was ignored."
               }
            }
         }
         shuffle_dimension = "rows"
      }
   }

   raw_alpha <- NA
   if(("LDA_VEM" %in% class(x[[1]])) | ("LDA_Gibbs" %in% class(x[[1]]))) {
      if(!(type %in% c("wt","td"))) {
         stop("If x is a list of S4 objects outputted from topicmodels::LDA(), then you\nmust specify the type argument as either 'wt' (to assess the reliability of word\nallocations to different topics) or 'td' (to assess the reliability of topic\nallocations to different documents).")
      }
      if(!shuffle) {
         warning("If x is a list of S4 objects outputted from topicmodels::LDA(), then each of\nthose S4 objects may have its distributions in a different order due to the\nrandom seeds used to conduct the analyses. It is strongly recommended that you\nset the shuffle argument to TRUE when assessing reliability for this data set.")
         for(i in 1:length(warnings)) {
            warnings[[i]][length(warnings[[i]])+1] <- "If x is a list of S4 objects outputted from topicmodels::LDA(), then each of\n   those S4 objects may have its distributions in a different order due to the\n   random seeds used to conduct the analyses. It is strongly recommended that\n   you set the shuffle argument to TRUE when assessing reliability for this data\n   set."
         }
      }
      if(type=="td") {
         if(identical(all.equal(alpha,NA),TRUE)) { #If the alpha values from topicmodels::LDA() will be useful, retain them for later use
            raw_alpha <- vector("list",length(x))
            for(i in 1:length(raw_alpha)) {
               raw_alpha[[i]] <- slot(x[[i]], "alpha")
            }
         }
         if(different_documents) { #If different documents were used in different cross-validation iterations, then in any iteration for which a given document is absent, all of its allocations must be set to NA (since setting them to 0 would compromise reliability and would also not constitute a valid Dirichlet distribution)
            warning("If you are evaluating the allocations of topics to documents, but different\ncross-validation iterations comprise different documents, then in any iteration\nfor which a given document is absent, all of its allocations must be set to NA.\nIf some documents appear in a small number of the cross-validation iterations\nthat you performed, then the reliability calculation for those documents may be\nunstable. These results should be interpreted with caution.")
            for(i in 1:length(warnings)) {
               warnings[[i]][length(warnings[[i]])+1] <- "If you are evaluating the allocations of topics to documents, but different\n   cross-validation iterations comprise different documents, then in any\n   iteration for which a given document is absent, all of its allocations must\n   be set to NA. If some documents appear in a small number of the\n   cross-validation iterations that you performed, then the reliability\n   calculation for those documents may be unstable. These results should be\n   interpreted with caution."
            }
            names <- character(0)
            for(i in 1:length(x)) {
               names <- union(names, slot(x[[i]], "documents"))
            }
            if(length(names) == 0) {
               stop("The documents were not identified by unique names, so different_documents\ncannot be TRUE.")
            }
            named_data <- matrix(0,0,length(names))
            colnames(named_data) <- names
            for(i in 1:length(x)) {
               temp_data <- t(slot(x[[i]], "gamma"))
               colnames(temp_data) <- slot(x[[i]], "documents")
               x[[i]] <- t(plyr::rbind.fill.matrix(named_data, temp_data))
               x[[i]][is.na(x[[i]])] <- 1/ncol(x[[1]])
               rownames(x[[i]]) <- NULL
            }
         } else {
            for(i in 1:length(x)) {
               x[[i]] <- slot(x[[i]], "gamma") #Each list entry becomes an M x K double vector, with each row summing to 1
            }
         }
      }
      if(type=="wt") {
         if(different_documents) { #If different documents were used in different cross-validation iterations, then in any iteration for which a given word is absent, its allocation must be set to 0
            names <- character(0)
            for(i in 1:length(x)) {
               names <- union(names, slot(x[[i]], "terms"))
            }
            named_data <- matrix(0,0,length(names))
            colnames(named_data) <- names
            for(i in 1:length(x)) {
               temp_data <- exp(slot(x[[i]], "beta"))
               colnames(temp_data) <- slot(x[[i]], "terms")
               x[[i]] <- plyr::rbind.fill.matrix(named_data, temp_data)
               x[[i]][is.na(x[[i]])] <- zero
               colnames(x[[i]]) <- NULL
            }
         } else {
            namelist <- list()
            for(i in 1:length(x)) {
               namelist[[i]] <- slot(x[[i]], "terms")
            }
            for(i in 1:(length(namelist)-1)) {
               for(j in (i+1):length(namelist)) {
                  if(!identical(all.equal(namelist[[i]], namelist[[j]]), TRUE)) {
                     stop("Different documents appear in different cross-validation iterations. You\nmust set different_documents=TRUE.")
                  }
               }
            }
            for(i in 1:length(x)) {
               x[[i]] <- exp(slot(x[[i]], "beta"))
            }
         }
      }

      if(!symmetric_alpha && (sampling_method != "nonparametric") && identical(all.equal(alpha,NA),TRUE)) {
         warning("If x is a list of S4 objects outputted from topicmodels::LDA(), then you are\nlikely attempting to assess the reliability of LDA results. In most LDAs, the\nprior distribution of words on topics or of topics on documents is symmetrical.\nIf that is the case for you, then you should set symmetric_alpha=TRUE. If you\ndid not use a symmetrical prior distribution, then you should keep\nsymmetric_alpha=FALSE. However, estimating a non-symmetrical prior distribution\nmay be extraordinarily slow, particularly for larger data sets, and this\nestimation is not always stable, which can affect your results. If you are\nworking with a non-symmetrical prior, then if at all possible, it is strongly\nrecommended that you specify the concentration parameters of the Dirichlet\ndistribution using the alpha parameter. Alternatively, consider setting\nsampling_method='nonparametric' in order to use nonparametric bootstrapping to\nestimate the expected differences.")
         for(i in 1:length(warnings)) {
            warnings[[i]][length(warnings[[i]])+1] <- "If x is a list of S4 objects outputted from topicmodels::LDA(), then you are\n   likely attempting to assess the reliability of LDA results. In most LDAs, the\n   prior distribution of words on topics or of topics on documents is\n   symmetrical. If that is the case for you, then you should set\n   symmetric_alpha=TRUE. If you did not use a symmetrical prior distribution,\n   then you should keep symmetric_alpha=FALSE. However, estimating a\n   non-symmetrical prior distribution may be extraordinarily slow, particularly\n   for larger data sets, and this estimation is not always stable, which can\n   affect your results. If you are working with a non-symmetrical prior, then if\n   at all possible, it is strongly recommended that you specify the\n   concentration parameters of the Dirichlet distribution using the alpha\n   parameter. Alternatively, consider setting sampling_method='nonparametric' in\n   order to use nonparametric bootstrapping to estimate the expected\n   differences."
         }
      }
   }

   for(i in 1:(length(x)-1)) {
      for(j in (i+1):(length(x))) {
         if((nrow(x[[i]]) != nrow(x[[j]])) && (ncol(x[[i]]) != ncol(x[[j]]))) {
            stop("Different cross-validation iterations have different dimensions.")
         }
         if(nrow(x[[i]]) != nrow(x[[j]])) {
            stop("Different cross-validation iterations have different numbers of rows.")
         }
         if(ncol(x[[i]]) != ncol(x[[j]])) {
            stop("Different cross-validation iterations have different numbers of columns.")
         }
      }
   }

   if(shuffle && shuffle_dimension=="rows" && sampling_method=="nonparametric") {
      warning("When shuffle=TRUE, shuffle_dimension='rows', and\nsampling_method='nonparametric', the expected differences are estimated by\nbootstrapping individual cells rather than entire rows. This can result in\nobservations whose support differs from the original distribution, which may\nbias the observation reordering procedure and the resulting estimates of\nexpected differences. The results should be interpreted with caution.")
      for(i in 1:length(warnings)) {
         warnings[[i]][length(warnings[[i]])+1] <- "When shuffle=TRUE, shuffle_dimension='rows', and\n   sampling_method='nonparametric', the expected differences are estimated by\n   bootstrapping individual cells rather than entire rows. This can result in\n   observations whose support differs from the original distribution, which may\n   bias the observation reordering procedure and the resulting estimates of\n   expected differences. The results should be interpreted with caution."
      }
   }

   #Ensure that data are Dirichlet-distributed
   largest_deviation <- 0
   for(i in 1:length(x)) {
      if((max(x[[i]], na.rm=TRUE) > 1) | (min(x[[i]], na.rm=TRUE) < 0)) {
         stop("brittnu() is designed to compute Britt's nu for Dirichlet-distributed\ndata. All values must fall between 0 and 1, and all rows must sum to 1.\nAt least one observaton in your data set falls outside the range from 0 to 1.\nIf your data were not drawn from a Dirichlet distribution, consider running\nbrittnu::euclidkrip() to compute Euclidean Krippendorff's alpha instead.")
      }
      row_sums <- rowSums(x[[i]])
      current_largest_deviation <- max(abs(row_sums-1))
      largest_deviation <- max(current_largest_deviation, largest_deviation)
   }
   if(largest_deviation > tol) {
      warning(paste0("brittnu() is designed to compute Britt's nu for Dirichlet-distributed data, with\nall rows summing to 1. At least one row in your data set appears to sum to a\nvalue that substantially deviates from 1 (largest deviation: ", format(largest_deviation, digits=6), ").\nCheck your data set to determine whether this is the result of a rounding error\nor whether your data set is non-Dirichlet distributed. If your data were not\ndrawn from a Dirichlet distribution, consider running brittnu::euclidkrip() to\ncompute Euclidean Krippendorff's alpha instead."))
      for(i in 1:length(warnings)) {
         warnings[[i]][length(warnings[[i]])+1] <- paste0("brittnu() is designed to compute Britt's nu for Dirichlet-distributed data,\n   with all rows summing to 1. At least one row in your data set appears to sum\n   to a value that substantially deviates from 1 (largest deviation: ", format(largest_deviation, digits=6), ").\n   Check your data set to determine whether this is the result of a rounding\n   error or whether your data set is non-Dirichlet distributed. If your data\n   were not drawn from a Dirichlet distribution, consider running\n   brittnu::euclidkrip() to compute Euclidean Krippendorff's alpha instead.")
      }
   }

   #Report warning when concentration parameters may have been shuffled
   if(shuffle==TRUE && shuffle_dimension=="columns" && sampling_method=="parametric" && (((!is.na(alpha)) && length(unique(alpha))>1) | (!symmetric_alpha))) {
      warning("When categories are reordered and the concentration parameters are not equal,\nthe reordered distributions may not match the reported or computed concentration\nparameters. If the reported/computed concentration parameters do not properly\nrepresent the data distribution, then a parametric method of computing the\nexpected differences may not yield accurate results. Consider setting\nsampling_method='nonparametric' in order to use nonparametric bootstrapping to\nestimate the expected differences.")
      for(i in 1:length(warnings)) {
         warnings[[i]][length(warnings[[i]])+1] <- "When categories are reordered and the concentration parameters are not equal,\n   the reordered distributions may not match the reported or computed\n   concentration parameters. If the reported/computed concentration parameters\n   do not properly represent the data distribution, then a parametric method of\n   computing the expected differences may not yield accurate results. Consider\n   setting sampling_method='nonparametric' in order to use nonparametric\n   bootstrapping to estimate the expected differences."
      }
   }

   #Report warning when lower_bound is used
   if(lower_bound) {
      warning("When lower_bound=TRUE, a lower bound is computed for all multiple-observation\nreliabilities. However, the reliabilities that are reported for individual\nobservations may not each be true lower bounds. This does not affect the\nmultiple-distribution reliabilities, but the single-distribution reliabilities\nshould be interpreted with caution.")
      warnings[[1]][length(warnings[[1]])+1] <- "When lower_bound=TRUE, a lower bound is computed for all multiple-observation\n   reliabilities. However, the reliabilities that are reported for individual\n   observations may not each be true lower bounds. These reported reliabilities\n   should be interpreted with caution."
      warnings[[2]][length(warnings[[2]])+1] <- "When lower_bound=TRUE, a lower bound is computed for all multiple-observation\n   reliabilities. However, the reliabilities that are reported for individual\n   observations may not each be true lower bounds. These reported reliabilities\n   should be interpreted with caution."
      warnings[[5]][length(warnings[[5]])+1] <- "When lower_bound=TRUE, a lower bound is computed for all multiple-observation\n   reliabilities. However, the reliabilities that are reported for individual\n   observations may not each be true lower bounds. This does not affect the\n   multiple-distribution reliabilities, but the single-distribution\n   reliabilities should be interpreted with caution."
   }

   #Shuffle the distributions/categories if necessary
   if(shuffle) {
      #Verify that the user is not trying to set distinct concentration parameters and then shuffle the elements within distributions (such that the elements and their concentration parameters will become mismatched)
      if(shuffle_dimension=="columns") {
         if(!identical(all.equal(alpha,NA),TRUE)) { #If alpha is not NA...
            if(length(alpha) > 1) {
               for(i in 2:length(alpha)) {
                  if(!identical(all.equal(alpha[[i]],alpha[[1]]),TRUE)) { #If any distributions have different sets of a priori alphas, the distributions cannot be shuffled
                     stop("If categories are allowed to be reordered, then alpha must be set to NA\nor a list of vectors in which all values with the same vector index are\nidentical. Distinct sets of alpha levels corresponding to different\ndistributions will not be retained when those distributions are shuffled.")
                  }
               }
            }
         }
      }
      shuffle_results <- shuffle_distributions(x, verbose, shuffle_method=shuffle_method, clustering_method=clustering_method, shuffle_dimension=shuffle_dimension, slow_clustering=slow_clustering)
      x <- shuffle_results[[1]] #Change the data object to be the newly shuffled data set
      assignments <- shuffle_results[[2]]
      rm(shuffle_results)
   } else {
      assignments = NA
   }

   #Obtain concentration parameters if they are needed
   if(sampling_method == "parametric") { #If this condition is true, then concentration parameters will be used later in the function
      if(identical(all.equal(alpha,NA),TRUE)) { #If concentration parameters have not already been specified, they must be obtained
         if(identical(all.equal(type,"td"),TRUE)) { #Type of reliability: topic-document
            if(!identical(all.equal(raw_alpha,NA),TRUE)) { #If this object is not NA, then topic-document alphas were already extracted from the results of topicmodels::LDA()
               joint_alpha <- lapply(1:nrow(x[[1]]), function(i) rep(mean(unlist(raw_alpha)),dim(x[[1]])[2]))
               if(pairwise) {
                  pairwise_alpha <- vector("list",(length(x)-1))
                  if(estimate_pairwise_alpha_from_joint) {
                     alphaarray <- t(sapply(1:(dim(x[[1]])[1]), function(k) rep(mean(unlist(raw_alpha)),dim(x[[1]])[2]), simplify="array"))
                     for(i in 1:(length(pairwise_alpha))) {
                        pairwise_alpha[[i]] <- vector("list",length(x))
                        for(j in i:length(pairwise_alpha[[i]])) {
                           pairwise_alpha[[i]][[j]] <- alphaarray
                        }
                     }
                  } else {
                     for(i in 1:(length(pairwise_alpha))) {
                        pairwise_alpha[[i]] <- vector("list",length(x))
                        for(j in i:length(pairwise_alpha[[i]])) {
                           pairwise_alpha[[i]][[j]] <- t(sapply(1:(dim(x[[1]])[1]), function(k) rep(((raw_alpha[[i]]+raw_alpha[[j]])/2),dim(x[[1]])[2]), simplify="array"))
                        }
                     }
                  }
               }
            } else { #Concentration parameters must be estimated from the data
               alphas <- estimate_alpha_from_data(x, pairwise, estimate_pairwise_alpha_from_joint, symmetric_alpha, convcrit=convcrit, maxit=maxit,
                                                  progress=progress)
               joint_alpha <- alphas[[1]]
               if(pairwise) {
                  pairwise_alpha <- alphas[[2]]
               }
            }
         } else { #Word-topic concentration parameters are not given by topicmodels::LDA() and therefore must be estimated from the data
            alphas <- estimate_alpha_from_data(x, pairwise, estimate_pairwise_alpha_from_joint, symmetric_alpha, convcrit=convcrit, maxit=maxit,
                                               progress=progress)
            joint_alpha <- alphas[[1]]
            if(pairwise) {
               pairwise_alpha <- alphas[[2]]
            }
            rm(alphas)
         }
      } else {
         joint_alpha <- alpha
         if(pairwise) {
            pairwise_alpha <- vector("list",(length(x)-1))
            alphaarray <- do.call("rbind", alpha)
            for(i in 1:length(pairwise_alpha)) {
               pairwise_alpha[[i]] <- vector("list",length(x))
               for(j in 1:length(pairwise_alpha[[i]])) {
                  pairwise_alpha[[i]][[j]] <- alphaarray
               }
            }
         }
      }
   } else {
      joint_alpha <- NA
      pairwise_alpha <- NA
   }

   #Get observed and expected differences for each individual distribution
   #First, compute each D_o
   differences <- compute_observed_differences(x)
   do_single_joint <- differences[[1]]
   do_single_pairwise <- differences[[2]]

   if(shuffle | force_bootstrapping) { #Use the modified formula to compute the joint D_e for each distribution and the pairwise D_e for each distribution and pair of iterations
      expectations <- estimate_single_expectation(x, joint_alpha, pairwise_alpha, pairwise, shuffle, shuffle_method, shuffle_dimension, clustering_method, slow_clustering, samples, sampling_method, lower_bound, verbose)
   } else { #Use the standard formula to compute the joint D_e for each distribution and the pairwise D_e for each distribution and pair of iterations
      expectations <- compute_single_expectation(x, joint_alpha, pairwise_alpha, pairwise, samples, sampling_method, verbose)
   }
   de_single_joint <- expectations[[1]]
   if(pairwise) {
      de_single_pairwise <- expectations[[2]]
   }

   if(verbose) {
      print("Setting up the output object")
      utils::flush.console()
   }

   reliability <- vector("list",7) #Will contain up to four types of reliability coefficients:
                                   #1. single-observation joint reliability
                                   #2. single-observation pairwise reliability
                                   #3. multiple-observation joint reliability
                                   #4. multiple-observation pairwise reliability
                                   #5. assignments matrix
                                   #6. warning messages that will be printed for summary() calls
                                   #7. type of reliability ("wt", "td", NA)

   reliability[[1]] <- (1 - (do_single_joint/de_single_joint)) #Compute joint single-distribution reliability
   if(pairwise) {
      reliability[[2]] <- do_single_pairwise #Compute pairwise single-distribution reliability
      for(i in 1:(length(x)-1)) { #For each iteration i...
         for(j in (i+1):length(x)) { #And for each fellow iteration j...
            reliability[[2]][[i]][[j]] <- (1 - (do_single_pairwise[[i]][[j]]/de_single_pairwise[[i]][[j]]))
         }
      }
   } else {
      reliability[[2]] <- "Pairwise reliability not computed."
      warnings[[2]] <- character(0)
   }
   reliability[[3]] <- (1 - (sum(do_single_joint)/sum(de_single_joint))) #Compute joint multiple-distribution reliability
   if(pairwise) {
      reliability[[4]] <- do_single_pairwise #Compute pairwise multiple-distribution reliability
      for(i in 1:(length(x)-1)) { #For each iteration i...
         for(j in (i+1):length(x)) { #And for each fellow iteration j...
            reliability[[4]][[i]][[j]] <- (1 - (sum(do_single_pairwise[[i]][[j]])/sum(de_single_pairwise[[i]][[j]])))
         }
      }
   } else {
      reliability[[4]] <- "Pairwise reliability not computed."
      warnings[[4]] <- character(0)
   }
   reliability[[5]] <- label_assignments(assignments, type)
   reliability[[6]] <- warnings
   reliability[[7]] <- type

   names(reliability) <- c("singleomnibus", "singlepairwise", "multipleomnibus", "multiplepairwise", "assignments", "warnings", "type")
   class(reliability[[1]]) <- "brittnu_rel_element"
   class(reliability[[2]]) <- "brittnu_rel_element"
   class(reliability[[3]]) <- "brittnu_rel_element"
   class(reliability[[4]]) <- "brittnu_rel_element"
   class(reliability) <- "brittnu_rel"

   return(reliability)
}

#' Euclidean Krippendorff's Alpha
#'
#' @description This function computes Euclidean Krippendorff's alpha for data
#' sets composed of compositional or Euclidean data, as described by Britt
#' (under review).
#' 
#' @details One of the most important uses for Euclidean Krippendorff's alpha is
#' to assess the reliability of the allocations of words to topics or of topics
#' to documents via topic modeling techniques that do not necessarily adhere to
#' a Dirichlet distribution. This includes, for instance, Dirichlet multinomial
#' mixture models, which are commonly used for short text topic modeling.
#'
#' Crucially, different topic modeling cross-validation iterations, which
#' effectively represent distinct raters of the same data set, may result in the
#' same topics appearing in a different sequence. As such, those topics, which
#' may represent either the rows or the columns of the data set, often need to
#' be reordered in order to yield an optimal fit.
#'
#' In practice, it is generally infeasible to assess every possible combination
#' of topics across all raters. As such, \link{euclidkrip} provides three
#' methods of converging toward an optimal topic order for each rater. The
#' default, \code{shuffle_method="rotational"}, takes the provided sequence and
#' swaps pairs of categories until no further swaps would further improve the
#' fit. \code{shuffle_method="agglomerative"} and
#' \code{shuffle_method="divisive"} instead perform hierarchical cluster
#' analyses of all topics, with the added restriction that two topics
#' constructed by the same rater may not appear in the same cluster. Notably,
#' the \code{"agglomerative"} option can take excessive time in some cases,
#' while the \code{"divisive"} option forms clusters based on the macro-level
#' dynamics of the data and may therefore yield weakly matched sets of
#' individual topics compared to the other available options.
#'
#' Regardless of whether the rows or columns of the data set must be reordered,
#' when computing Euclidean Krippendorff's alpha, it is common to compute two
#' separate coefficients in this family: Euclidean Krippendorff's alpha for all
#' raters and for pairs of raters. Unlike Britt's nu, both of these Euclidean
#' Krippendorff's alpha measures yield a single reliability score for all
#' observations, as the formula used by Euclidean Krippendorff's alpha to
#' compute the expected differences between raters does not produce separate
#' results for each individual observation. However, if the reliability of each
#' observation is desired, then nonparametric bootstrapping may be used to
#' estimate those coefficients. This is automatically done when
#' \code{shuffle==TRUE}, and it is also implemented when
#' \code{force_bootstrapping==TRUE}. In either of those cases, the
#' single-observation reliability values will be reported alongside the
#' multiple-observation reliability values that \link{euclidkrip} already
#' provides.
#'
#' By default, \link{euclidkrip} provides reliability coefficients for all
#' raters and for pairs of raters. To reduce the time and memory required, the
#' pairwise versions may optionally be omitted from the computation by
#' specifying \code{pairwise=FALSE}.
#' 
#' Additional usage notes:
#'
#' 1. \code{x} must be a list of 2D double vectors, and each row of each list
#' element should be a single observation from a Dirichlet distribution (e.g.,
#' an LDA topic) while each column should represent a category within that
#' distribution (e.g., a word).
#'
#' 2. When \code{shuffle==TRUE}, each element in \code{x} comprises a large
#' number of topics, and/or \code{samples} is large, setting
#' \code{lower_bound=TRUE} may sometimes be useful in order to eliminate the
#' need to reorder a large number of topics in numerous bootstrapped samples.
#' You may also wish to set \code{verbose=TRUE} to receive periodic progress
#' updates and ensure that the computation is proceeding as expected.
#'
#' 3. When \code{shuffle==TRUE}, the topics in \code{x} are reordered. This is
#' described in the matrix of reordered topics, which is provided as one of the
#' elements of the list returned by \link{euclidkrip} and can be viewed using
#' \code{summary()} on that object. Each row of this matrix indicates the manner
#' in which the topics were reordered. For instance, if the first row is
#' c(3,1,2), that means that for the first rater, the third topic was moved to
#' the first position, the first topic was moved to the second position, and the
#' second topic was moved to the third position.
#'
#' @param x A list such that each element is a 2D double vector with
#'   observations as the rows and each category within each observation as the
#'   columns
#' @param pairwise A boolean value indicating whether pairwise reliability
#'   between individual raters should be computed; if \code{FALSE}, pairwise
#'   reliability will be ignored in order to reduce the time and memory
#'   complexity of the computation
#' @param force_bootstrapping A boolean value indicating whether bootstrapping
#'   should be used to estimate expected differences rather than exactly
#'   computing them even if \code{shuffle==FALSE}
#' @param shuffle A boolean value indicating whether or not the topics may have
#'   been shuffled into different sequences by each rater; if this is
#'   \code{TRUE} and \code{shuffle_dimension="rows"}, then the observations
#'   (rows) corresponding to each rater will be reordered to achieve a (local)
#'   optimal fit, whereas if this is \code{TRUE} and
#'   \code{shuffle_dimension="columns"}, the categories (columns) will be
#'   reordered instead
#' @param shuffle_method A string indicating what method (\code{"rotational"},
#'   \code{"agglomerative"}, or \code{"divisive"}) will be used to reorder
#'   topics
#' @param shuffle_dimension A string indicating whether \code{"rows"} or
#'   \code{"columns"} should be reordered; this argument only has an effect if
#'   \code{shuffle==TRUE} and \code{type==NA}
#' @param clustering_method A string indicating the criterion that will be used
#'   to merge or divide clusters if \code{shuffle==TRUE} and either
#'   \code{shuffle_method=="agglomerative"} or
#'   \code{shuffle_method=="divisive"}; permitted values are \code{"average"},
#'   \code{"minimum"}, \code{"maximum"}, \code{"median"}, \code{"centroid"}, and
#'   \code{"wald"}
#' @param slow_clustering A boolean value indicating whether the distances
#'   between clusters should be recalculated after each individual observation
#'   is added to the new cluster (\code{TRUE}), which can improve the
#'   cohesiveness of the resulting cluster, or whether all observations should
#'   be added to the new cluster without recomputing the distances between
#'   clusters after every addition (\code{FALSE}); this argument only has an
#'   effect if \code{shuffle==TRUE} and \code{shuffle_method=="divisive"}
#' @param samples An integer value indicating how many samples to use to
#'   estimate expected differences when \code{shuffle==TRUE}
#' @param lower_bound A boolean value indicating whether the lower bound of the
#'   expected differences (based on i.i.d. beta distributions) should be used
#'   rather than shuffling Dirichlet distributions to estimate the exact
#'   difference in every sample when \code{shuffle==TRUE}
#' @param verbose A boolean value indicating whether \link{euclidkrip} and its
#'   helper functions should provide progress updates beyond the estimation of
#'   concentration parameters
#' @return A list of class \code{euclidkrip_rel} containing six elements:
#'   Euclidean Krippendorff's alpha for each observation and all raters (as a
#'   numeric vector indicating the reliability for each observation), Euclidean
#'   Krippendorff's alpha for each observation and pairs of raters (as a list
#'   containing lists containing numeric vectors indicating the reliability of
#'   each observation for each pair of raters, e.g., the second element of the
#'   first list contains the reliability for raters 1 and 2), Euclidean
#'   Krippendorff's alpha for all observations and all raters (as a numeric
#'   value), Euclidean Krippendorff's alpha for all observations and pairs of
#'   raters (as a list containing lists containing numeric values, e.g., the
#'   second element of the first list contains the reliability for raters 1 and
#'   2), the matrix of reordered topics for each rater (if
#'   \code{shuffle==TRUE}), and any warnings raised
#' @importFrom "utils" "flush.console"
#' @section References:
#'   Britt, B. C. (under review). Interrater reliability for compositional,
#'   Euclidean, and Dirichlet Data.
#' @examples
#' rater1 <- rbind(c(0.80,0.05,0.15), c(0.10,0.85,0.05),
#'                 c(0.05,0.05,0.90), c(0.85,0.10,0.05))
#' rater2 <- rbind(c(0.10,0.85,0.05), c(0.10,0.10,0.80),
#'                 c(0.85,0.05,0.10), c(0.15,0.80,0.05))
#' data <- list(rater1, rater2)
#' rel <- euclidkrip(data, shuffle=TRUE, shuffle_method="agglomerative",
#'                   shuffle_dimension="columns", clustering_method="average",
#'                   samples=500, verbose=TRUE)
#' summary(rel)
#' summary(rel, element="all")
#' @export
#' euclidkrip

euclidkrip <- function(x, pairwise=TRUE, force_bootstrapping=FALSE, shuffle=FALSE, shuffle_method="rotational", shuffle_dimension="rows",
                     clustering_method="average", slow_clustering=FALSE, samples=1000, lower_bound=FALSE, verbose=FALSE) {
   if(("LDA_VEM" %in% class(x[[1]])) | ("LDA_Gibbs" %in% class(x[[1]]))) {
      stop("x must be a list of double vectors. If x represents results obtained via\nlatent Dirichlet allocation (LDA), then it is recommended that Britt's nu be\ncomputed instead by calling brittnu::brittnu(). If you wish to compute\nEuclidean Krippendorff's alpha for LDA results, then you will need to extract\nthe relevant data from each LDA model, such as by running\n\n   for(i in 1:length(x)) {\n      x[[i]] <- slot(x[[i]], 'gamma')\n   }\n\nto prepare x to compute the reliability of topic-document allocations,\nor by running\n\n   for(i in 1:length(x)) {\n      x[[i]] <- exp(slot(x[[i]], 'beta'))\n   }\n\nif you seek to evaluate the reliability of word-topic allocations.")
   }

   warnings <- vector("list", 5) #One vector for each of the individual reliability types, and a separate (fifth) vector for when all reliabilities are printed
   for(i in 1:length(warnings)) {
      warnings[[i]] <- vector("character")
   }

   if(shuffle && shuffle_dimension=="rows") {
      warning("When shuffle=TRUE and shuffle_dimension='rows', the expected differences are\nestimated by bootstrapping individual cells rather than entire rows. This can\nresult in observations whose support differs from the original distribution,\nwhich may bias the observation reordering procedure and the resulting estimates\nof expected differences. The results should be interpreted with caution.")
      for(i in 1:length(warnings)) {
         warnings[[i]][length(warnings[[i]])+1] <- "When shuffle=TRUE and shuffle_dimension='rows', the expected differences are\n   estimated by bootstrapping individual cells rather than entire rows. This can\n   result in observations whose support differs from the original distribution,\n   which may bias the observation reordering procedure and the resulting\n   estimates of expected differences. The results should be interpreted with\n   caution."
      }
   }

   if(shuffle==FALSE & lower_bound==TRUE) {
      warning("When shuffle=FALSE, there is no need to compute the lower bound for Euclidean\nKrippendorff's alpha that could occur based on category or observation\nreordering, so lower_bound=FALSE was ignored.")
      for(i in 1:length(warnings)) {
         warnings[[i]][length(warnings[[i]])+1] <- "When shuffle=FALSE, there is no need to compute the lower bound for Euclidean\n   Krippendorff's alpha that could occur based on category or observation\n   reordering, so lower_bound=FALSE was ignored."
      }
      lower_bound = FALSE
   }

   if(shuffle) {
      shuffle_results <- shuffle_distributions(x, verbose, shuffle_method=shuffle_method, clustering_method=clustering_method, slow_clustering=slow_clustering, shuffle_dimension=shuffle_dimension)
      x <- shuffle_results[[1]] #Change the data object to be the newly shuffled data set
      assignments <- shuffle_results[[2]]
      rm(shuffle_results)
   } else {
      assignments = NA
   }

   #Get observed and expected differences for each individual distribution
   #First, compute each D_o
   differences <- compute_observed_differences(x)
   do_joint <- differences[[1]]
   do_pairwise <- differences[[2]]

   #Next, prepare a data structure for the expected differences
   de_joint <- 0
   de_pairwise <- vector("list",length(x))
   for(i in 1:length(do_pairwise)) { #For each iteration i...
      de_pairwise[[i]] <- vector("list",length(x))
      for(j in 1:length(do_pairwise[[i]])) { #And for each corresponding iteration j...
         de_pairwise[[i]][[j]] <- 0 #This value indicates the expected sum of squared differences between iterations i and j
      }
   }

   if(shuffle | force_bootstrapping) { #If the data were shuffled or if we're deliberately performing bootstrapping, then we need to compute the expected differences based on that rather than using the usual Euclidean Krippendorff's alpha formula
      expectations <- estimate_single_expectation(x, joint_alpha=NA, pairwise_alpha=NA, pairwise=TRUE, shuffle=shuffle, shuffle_method=shuffle_method, shuffle_dimension=shuffle_dimension, clustering_method=clustering_method, slow_clustering, samples=samples, sampling_method="nonparametric", lower_bound=lower_bound, verbose=verbose)

      de_joint <- expectations[[1]]
      if(pairwise) {
         de_pairwise <- expectations[[2]]
      }
   } else {
      for(i in 1:(length(x)-1)) {
         for(j in (i+1):length(x)) {
            if(verbose) {
               print(paste0("Computing expected differences: Raters ", i, " and ", j))
               utils::flush.console()
            }
            for(k in 1:nrow(x[[i]])) {
               for(l in 1:nrow(x[[j]])) {
                  de_pairwise[[i]][[j]] <- de_pairwise[[i]][[j]] + (((x[[i]][k,] - x[[j]][l,])^2) / nrow(x[[i]]))
               }
            }
            de_pairwise[[i]][[j]] <- de_pairwise[[i]][[j]] / (nrow(x[[i]]) * (nrow(x[[j]])))
            de_joint <- de_joint + de_pairwise[[i]][[j]]
         }
      }
      de_joint <- de_joint * 2 / (length(x)*(length(x)-1))
   }

   if(verbose) {
      print("Setting up the output object")
      utils::flush.console()
   }

   reliability <- vector("list",6) #Will contain up to four types of reliability coefficients:
                                   #1. single-observation joint reliability
                                   #2. single-observation pairwise reliability
                                   #3. multiple-observation joint reliability
                                   #4. multiple-observation pairwise reliability
                                   #5. assignments matrix
                                   #6. warning messages that will be printed for summary() calls

   if(shuffle | force_bootstrapping) {
      reliability[[1]] <- (1 - (do_joint/de_joint)) #Compute joint single-observation reliability
      if(pairwise) {
         reliability[[2]] <- do_pairwise #Compute pairwise single-observation reliability
         for(i in 1:(length(x)-1)) { #For each iteration i...
            for(j in (i+1):length(x)) { #And for each fellow iteration j...
               reliability[[2]][[i]][[j]] <- (1 - (do_pairwise[[i]][[j]]/de_pairwise[[i]][[j]]))
            }
         }
      } else {
         reliability[[2]] <- "Pairwise reliability not computed."
         warnings[[2]] <- character(0)
      }
   } else {
      reliability[[1]] <- "Single-observation reliability can only be estimated for Euclidean Krippendorff's alpha via bootstrapping. If you want to assess the reliability of each individual observation, set force_bootstrapping=TRUE."
      reliability[[2]] <- "Single-observation reliability can only be estimated for Euclidean Krippendorff's alpha via bootstrapping. If you want to assess the reliability of each individual observation, set force_bootstrapping=TRUE."
      warnings[[1]] <- character(0)
      warnings[[2]] <- character(0)
   }
   reliability[[3]] <- (1 - (sum(do_joint)/sum(de_joint))) #Compute joint multiple-observation reliability
   if(pairwise) {
      reliability[[4]] <- do_pairwise #Compute pairwise multiple-observation reliability
      for(i in 1:(length(x)-1)) { #For each iteration i...
         for(j in (i+1):length(x)) { #And for each fellow iteration j...
            reliability[[4]][[i]][[j]] <- (1 - (sum(do_pairwise[[i]][[j]])/sum(de_pairwise[[i]][[j]])))
         }
      }
   } else {
      reliability[[4]] <- "Pairwise reliability not computed."
      warnings[[4]] <- character(0)
   }
   reliability[[5]] <- label_assignments(assignments, type=NA)
   reliability[[6]] <- warnings
   names(reliability) <- c("singleomnibus", "singlepairwise", "multipleomnibus", "multiplepairwise", "assignments", "warnings")
   class(reliability[[1]]) <- "euclidkrip_rel_element"
   class(reliability[[2]]) <- "euclidkrip_rel_element"
   class(reliability[[3]]) <- "euclidkrip_rel_element"
   class(reliability[[4]]) <- "euclidkrip_rel_element"
   class(reliability) <- "euclidkrip_rel"

   return(reliability)
}

#' Label Assignments Matrix
#'
#' @description This helper function adds labels to the assignments matrix based
#'   on whether the rows and columns represent documents and topics, topics and
#'   words, or something else, in order to facilitate more informative printing.
#'
#' @param assignments A matrix indicating the relationship between each
#'   reordered topic and its original position in the data
#' @param type A string indicating the type of reliability being assessed, with
#'   permitted values of \code{"wt"} (word-topic reliability), \code{"td"}
#'   (topic-document reliability), and NA
#' @return A matrix indicating the relationship between each reordered topic and
#'   its original position in the data, with the addition of row and column
#'   labels
#' @export
#' label_assignments

label_assignments <- function(assignments, type=NA) {
   if(length(assignments) <= 1) {
      return(assignments)
   }
   if(is.na(type)) {
      rownames(assignments) <- paste0("Row ", c(1:nrow(assignments)))
      colnames(assignments) <- paste0("Column ", c(1:ncol(assignments)))
      return(assignments)
   } else if(type=="td") {
      rownames(assignments) <- paste0("Document ", c(1:nrow(assignments)))
      colnames(assignments) <- paste0("Topic ", c(1:ncol(assignments)))
      return(assignments)      
   } else if(type=="wt") {
      rownames(assignments) <- paste0("Topic ", c(1:nrow(assignments)))
      colnames(assignments) <- paste0("Word ", c(1:ncol(assignments)))
      return(assignments)
   }
}

#' Compute Observed Differences
#'
#' @description This helper function computes the differences between all raters
#'   and between all pairs of raters in a given real or bootstrapped data set.
#'
#' @param x A list such that each element is a 2D double vector with
#'   observations as the rows and each category within each observation as the
#'   columns
#' @return A two-element list indicating the observed differences between
#'   observations for all raters and for each pair of raters, respectively
#' @export
#' compute_observed_differences

compute_observed_differences <- function(x) {
   do_joint <- 0
   do_pairwise <- vector("list",length(x)-1)
   for(i in 1:length(do_pairwise)) { #For each iteration i...
      do_pairwise[[i]] <- vector("list",length(x))
      for(j in 1:length(do_pairwise[[i]])) { #And for each corresponding iteration j...
         do_pairwise[[i]][[j]] <- vector("double") #This vector indicates the sum of squared differences between iterations i and j for each respective distribution
      }
   }
   for(i in 1:(length(do_pairwise))) { #Compute pairwise D_o value for iterations i and j
      for(j in (i+1):length(do_pairwise[[i]])) {
         do_pairwise[[i]][[j]] <- rowSums((x[[i]] - x[[j]])^2) / nrow(x[[i]])
         do_joint <- do_joint + do_pairwise[[i]][[j]]
      }
   }
   do_joint <- do_joint * 2 / (length(x) * (length(x)-1))
   return(list(do_joint, do_pairwise))
}

#' Compute Expected Joint Differences (Lower Bound)
#'
#' @description This helper function estimates the lower bound of the expected
#'   differences among all raters.
#'
#' @param x A list such that each element is a 2D double vector with
#'   observations as the rows and each category within each observation as the
#'   columns
#' @param shuffle_dimension A string indicating whether \code{"rows"} or
#'   \code{"columns"} were shuffled in the original data set
#' @param verbose A boolean value indicating whether to provide progress updates
#' @return A list of lists containing numeric vectors indicating the lower bound
#'   of the expected differences between all raters
#' @export
#' compute_expected_joint_differences_lower_bound

compute_expected_joint_differences_lower_bound <- function(x, shuffle_dimension, verbose) {
   if(shuffle_dimension=="rows") { #We need to evaluate the dimension along which values would actually be shuffled
      for(i in 1:length(x)) {
         x[[i]] <- t(x[[i]])
      }
   }
   joint_sum <- rep(0, nrow(x[[1]]))
   for(i in 1:(length(x)-1)) {
      for(j in (i+1):length(x)) {
         if(verbose) {
            print(paste0("Computing lower bound of expected joint differences: Raters ", i, " and ", j))
            utils::flush.console()
         }
         for(k in 1:ncol(x[[i]])) {
            best_diff <- Inf
            best_vector <- numeric(0)
            for(l in 1:ncol(x[[j]])) {
               current_diff <- sum((x[[i]][,k] - x[[j]][,l])^2)
               if(current_diff < best_diff) {
                  best_diff <- current_diff
                  best_vector <- (x[[i]][,k] - x[[j]][,l])^2
               }
            }
            joint_sum <- joint_sum + (best_vector / nrow(x[[i]]))
         }
      }
   }
   return(joint_sum * 2 / (length(x) * (length(x)-1)))
}

#' Compute Expected Pairwise Differences (Lower Bound)
#'
#' @description This helper function estimates the lower bound of the expected
#'   differences between pairs of raters.
#'
#' @param x A list such that each element is a 2D double vector with
#'   observations as the rows and each category within each observation as the
#'   columns
#' @param shuffle_dimension A string indicating whether \code{"rows"} or
#'   \code{"columns"} were shuffled in the original data set
#' @param verbose A boolean value indicating whether to provide progress updates
#' @return A list of lists containing numeric vectors indicating the lower bound
#'   of the expected differences between pairs of raters
#' @export
#' compute_expected_pairwise_differences_lower_bound

compute_expected_pairwise_differences_lower_bound <- function(x, shuffle_dimension, verbose) {
   if(shuffle_dimension=="rows") { #We need to evaluate the dimension along which values would actually be shuffled
      for(i in 1:length(x)) {
         x[[i]] <- t(x[[i]])
      }
   }
   pairwise_sum <- vector("list", length(x)-1)
   for(i in 1:(length(x)-1)) {
      pairwise_sum[[i]] <- vector("list", length(x))
      for(j in (i+1):length(x)) {
         if(verbose) {
            print(paste0("Computing lower bound of expected pairwise differences: Raters ", i, " and ", j))
            utils::flush.console()
         }
         pairwise_sum[[i]][[j]] <- rep(0, ncol(x[[1]]))
         for(k in 1:nrow(x[[i]])) {
            best_diff <- Inf
            best_vector <- numeric(0)
            for(l in 1:nrow(x[[j]])) {
               current_diff <- sum((x[[i]][k,] - x[[j]][l,])^2)
               if(current_diff < best_diff) {
                  best_diff <- current_diff
                  best_vector <- (x[[i]][k,] - x[[j]][l,])^2
               }
            }
            pairwise_sum[[i]][[j]] <- pairwise_sum[[i]][[j]] + (best_vector / nrow(x[[i]]))
         }
      }
   }
   return(pairwise_sum)
}

#' Generate Artificial Data (Nonparametric)
#'
#' @description This helper function uses nonparametric bootstrapping to
#'   generate an artificial data set from a provided data set
#'
#' @param x A list such that each element is a 2D double vector with
#'   observations as the rows and each category within each observation as the
#'   columns
#' @param shuffle A boolean value indicating whether or not the topics may have
#'   been shuffled into different sequences by each rater
#' @param shuffle_dimension A string indicating whether \code{"rows"} or
#'   \code{"columns"} were shuffled in the original data set; this argument only
#'   has an effect if \code{shuffle==TRUE}
#' @return A list of nonparametrically bootstrapped data such that each element
#'   is a 2D double vector with observations as the rows and each category
#'   within each observation as the columns
#' @export
#' generate_artificial_data_nonparametric

generate_artificial_data_nonparametric <- function(x, shuffle, shuffle_dimension) {
   sample_data <- vector("list", length(x))
   for(i in 1:length(x)) { #Randomly reorder the columns in each row
      sample_data[[i]] <- matrix(NA, nrow(x[[i]]), ncol(x[[i]]))
      if(is.na(shuffle_dimension)) {
         shuffle_dimension <- "columns" #If no shuffle_dimension was set, then set it to columns within this function so that it won't affect bootstrapped data generation
      }
      if(shuffle_dimension=="rows" && shuffle) { #If we're allowed to shuffle rows, then we need to bootstrap elements within each column
         for(j in 1:ncol(sample_data[[i]])) {
            row_numbers <- sample(1:nrow(x[[i]]), nrow(x[[i]]), replace=FALSE)
            sample_data[[i]][,j] <- x[[i]][row_numbers,j]
         }
      } else { #If we're not allowed to shuffle rows, then we need to bootstrap rows
         row_numbers <- sample(1:nrow(x[[i]]), nrow(x[[i]]), replace=TRUE)
         sample_data[[i]] <- x[[i]][row_numbers,]
      }
   }
   return(sample_data)
}

#' Generate Artificial Data (Parametric)
#'
#' @description This helper function uses parametric bootstrapping to generate
#'   an artificial data set from a provided data set
#'
#' @param x A list such that each element is a 2D double vector with
#'   observations as the rows and each category within each observation as the
#'   columns
#' @param pairwise_alpha A list of lists of vectors of concentration parameters
#'   representing all pairs of raters
#' @return A list of parametrically bootstrapped data such that each element is
#'   a 2D double vector with observations as the rows and each category within
#'   each observation as the columns
#' @export
#' generate_artificial_data_parametric

generate_artificial_data_parametric <- function(x, pairwise_alpha) {
   sample_data <- vector("list",length(x))

   for(i in 1:(length(sample_data)-1)) {
      sample_data[[i]] <- vector("list",length(x))
      for(j in (i+1):length(sample_data[[i]])) {
         sample_data[[i]][[j]] <- vector("list", 2)
         for(k in 1:length(sample_data[[i]][[j]])) {
            for(l in 1:nrow(pairwise_alpha[[i]][[j]])) {
               if(l==1) {
                  sample_data[[i]][[j]][[k]] <- rdirichlet(1, pairwise_alpha[[i]][[j]][l,])
               } else {
                  sample_data[[i]][[j]][[k]] <- rbind(sample_data[[i]][[j]][[k]], rdirichlet(1, pairwise_alpha[[i]][[j]][l,]))
               }
            }
         }
      }
   }
   return(sample_data)
}

#' Estimate Single Expectation
#'
#' @description This helper function uses bootstrapping to estimate the expected
#'   squared difference between observations from the same underlying
#'   distribution.
#'
#' @param x A list such that each element is a 2D double vector with
#'   observations as the rows and each category within each observation as the
#'   columns
#' @param joint_alpha A vector of concentration parameters representing all
#'   raters
#' @param pairwise_alpha A list of lists of vectors of concentration parameters
#'   representing all pairs of raters
#' @param pairwise A boolean value indicating whether pairwise differences
#'   between individual raters should be computed; if \code{FALSE}, pairwise
#'   differences will be ignored in order to reduce the time and memory
#'   complexity of the computation
#' @param shuffle A boolean value indicating whether or not the topics may have
#'   been shuffled into different sequences by each rater
#' @param shuffle_method A string indicating what method (\code{"rotational"},
#'   \code{"agglomerative"}, or \code{"divisive"}) will be used to reorder
#'   topics
#' @param shuffle_dimension A string indicating whether \code{"rows"} or
#'   \code{"columns"} should be reordered; this argument only has an effect if
#'   \code{shuffle==TRUE} and \code{type==NA}
#' @param clustering_method A string indicating the criterion that will be used
#'   to merge or divide clusters if \code{shuffle==TRUE} and either
#'   \code{shuffle_method=="agglomerative"} or
#'   \code{shuffle_method=="divisive"}; permitted values are \code{"average"},
#'   \code{"minimum"}, \code{"maximum"}, \code{"median"}, \code{"centroid"}, and
#'   \code{"wald"}
#' @param slow_clustering A boolean value indicating whether the distances
#'   between clusters should be recalculated after each individual observation
#'   is added to the new cluster (\code{TRUE}), which can improve the
#'   cohesiveness of the resulting cluster, or whether all observations should
#'   be added to the new cluster without recomputing the distances between
#'   clusters after every addition (\code{FALSE}); this argument only has an
#'   effect if \code{shuffle==TRUE} and \code{shuffle_method=="divisive"}
#' @param samples An integer value indicating how many samples to use to
#'   estimate expected differences
#' @param sampling_method A string indicating whether bootstrapped sampling
#'   should be performed using a \code{"parametric"} or \code{"nonparametric"}
#'   approach
#' @param lower_bound A boolean value indicating whether the lower bound of the
#'   expected differences (based on i.i.d. beta distributions) should be used
#'   rather than reordering Dirichlet distributions to estimate the exact
#'   difference in every sample when \code{shuffle==TRUE}
#' @param verbose A boolean value indicating whether to provide progress updates
#' @return A two-element list indicating the expected differences between
#'   observations for all raters and for each pair of raters, respectively
#' @export
#' estimate_single_expectation

estimate_single_expectation <- function(x, joint_alpha, pairwise_alpha, pairwise, shuffle=TRUE, shuffle_method="rotational", shuffle_dimension="rows", clustering_method="average", slow_clustering=FALSE, samples=1000, sampling_method="nonparametric", lower_bound=TRUE, verbose=FALSE) {
   single_joint_expectation <- vector("list", samples) #Expected difference for one distribution across all iterations

   for(i in 1:length(single_joint_expectation)) {
      #Generate artificial data
      single_joint_expectation[[i]] <- vector("list", length(x))
      if(verbose) {
         print(paste0("Bootstrapping data for omnibus expected difference: Creating sample ", i))
         utils::flush.console()
      }
      if(sampling_method=="nonparametric") {
         single_joint_expectation[[i]] <- generate_artificial_data_nonparametric(x, shuffle=TRUE, shuffle_dimension=shuffle_dimension)
      } else if(sampling_method=="parametric") {
         for(j in 1:length(joint_alpha)) {
            if(j==1) {
               for(k in 1:length(single_joint_expectation[[i]])) {
                  single_joint_expectation[[i]][[k]] <- rdirichlet(1, joint_alpha[[j]])
               }
            } else {
               for(k in 1:length(single_joint_expectation[[i]])) {
                  single_joint_expectation[[i]][[k]] <- rbind(single_joint_expectation[[i]][[k]], rdirichlet(1, joint_alpha[[j]]))
               }
            }
         }
      }

      #Compute expected differences for this sample
      if(shuffle) { #If we're allowed to shuffle observations or categories, then account for that when estimating expected differences
         if(lower_bound) { #Obtain a lower bound for D_e by comparing each category of each distribution against all possible counterparts and selecting the minimum rather than doing time-intensive shuffling to find the "correct" match
                           #Note: In some cases, individual distributions may have larger expected differences for the "lower bound" than a shuffling-based approach would have yielded
                           #However, this will yield the lower bound for the sum of all expected differences across all distributions
            single_joint_expectation[[i]] <- compute_expected_joint_differences_lower_bound(single_joint_expectation[[i]], shuffle_dimension, verbose)
         } else { #Shuffle all distributions rather than using the minimum categorywise match in every case
            single_joint_expectation[[i]] <- shuffle_distributions(single_joint_expectation[[i]], verbose, shuffle_method=shuffle_method, clustering_method=clustering_method, slow_clustering=slow_clustering, shuffle_dimension=shuffle_dimension)[[1]]
            single_joint_expectation[[i]] <- compute_observed_differences(single_joint_expectation[[i]])[[1]]
         }
      } else { #We cannot shuffle observations or categories, so don't adjust the observed differences
         single_joint_expectation[[i]] <- compute_observed_differences(single_joint_expectation[[i]])[[1]]
      }
   }
   single_joint_expectation <- Reduce(`+`, single_joint_expectation) / samples #Take the mean of all of the samples

   if(pairwise) {
      single_pairwise_sum <- vector("list", length(x)-1)
      for(i in 1:length(single_pairwise_sum)) {
         single_pairwise_sum[[i]] <- vector("list", length(x))
         for(j in 1:length(single_pairwise_sum[[i]])) {
            single_pairwise_sum[[i]][[j]] <- rep(0, length(x[[1]][,1]))
         }
      }
      single_pairwise_expectation <- vector("list", samples)
      if(sampling_method=="nonparametric") {
         for(h in 1:samples) {
            single_pairwise_expectation[[h]] <- vector("list",length(x)) #Expected difference for one distribution across pairs of iterations

            #Generate artificial data
            if(verbose) {
               print(paste0("Bootstrapping data for pairwise expected difference: Creating sample ", h))
               utils::flush.console()
            }
            single_pairwise_expectation[[h]] <- generate_artificial_data_nonparametric(x, shuffle=TRUE, shuffle_dimension=shuffle_dimension)

            #Compute pairwise expected difference
            if(shuffle) { #If we're allowed to shuffle observations or categories, then account for that when estimating expected differences
               if(lower_bound) { #Obtain a lower bound for D_e by comparing each category of each distribution against all possible counterparts and selecting the minimum rather than doing time-intensive shuffling to find the "correct" match
                                 #Note: In some cases, individual distributions may have larger expected differences for the "lower bound" than a shuffling-based approach would have yielded
                                 #However, this will yield the lower bound for the sum of all differences across all distributions
                  single_pairwise_expectation[[h]] <- compute_expected_pairwise_differences_lower_bound(single_pairwise_expectation[[h]], shuffle_dimension, verbose)
               } else {
                  single_pairwise_expectation[[h]] <- shuffle_distributions(single_pairwise_expectation[[h]], verbose, shuffle_method=shuffle_method, clustering_method=clustering_method, slow_clustering=slow_clustering, shuffle_dimension=shuffle_dimension)[[1]]
                  single_pairwise_expectation[[h]] <- compute_observed_differences(single_pairwise_expectation[[h]])[[2]]
               }
            } else { #We cannot shuffle observations or categories, so don't adjust the observed differences
               single_pairwise_expectation[[h]] <- compute_observed_differences(single_pairwise_expectation[[h]])[[2]]
            }
            for(i in 1:length(single_pairwise_expectation[[h]])) {
               for(j in (i+1):length(single_pairwise_expectation[[h]][[i]])) {
                  single_pairwise_sum[[i]][[j]] <- single_pairwise_sum[[i]][[j]] + (single_pairwise_expectation[[h]][[i]][[j]] / samples)
               }
            }
         single_pairwise_expectation[[h]] <- NA
         }
         single_pairwise_expectation <- single_pairwise_sum
      } else if(sampling_method=="parametric") {
         for(h in 1:samples) {
            single_pairwise_expectation[[h]] <- vector("list",length(x)) #Expected difference for one distribution across pairs of iterations

            #Generate artificial data
            if(verbose) {
               print(paste0("Bootstrapping data for pairwise expected difference: Creating sample ", h))
               utils::flush.console()
            }
            single_pairwise_expectation[[h]] <- generate_artificial_data_parametric(x, pairwise_alpha)

            #Compute pairwise expected difference
            for(i in 1:(length(x)-1)) {
               for(j in (i+1):length(x)) {
                  if(lower_bound) {
                     single_pairwise_sum[[i]][[j]] <- single_pairwise_sum[[i]][[j]] + (compute_expected_pairwise_differences_lower_bound(single_pairwise_expectation[[h]][[i]][[j]], shuffle_dimension, verbose)[[1]][[2]] / samples)
                  } else {
                     single_pairwise_expectation[[h]][[i]][[j]] <- shuffle_distributions(single_pairwise_expectation[[h]][[i]][[j]], verbose, shuffle_method=shuffle_method, clustering_method=clustering_method, slow_clustering=slow_clustering, shuffle_dimension=shuffle_dimension)[[1]]
                     single_pairwise_sum[[i]][[j]] <- single_pairwise_sum[[i]][[j]] + (compute_observed_differences(single_pairwise_expectation[[h]][[i]][[j]])[[1]] / samples)
                  }
               }
            }
         }
         single_pairwise_expectation <- single_pairwise_sum
      }
   } else {
      single_pairwise_expectation <- NA
   }
   return(list(single_joint_expectation, single_pairwise_expectation))
}

#' Compute Single Expectation
#'
#' @description This helper function computes the expected squared difference
#'   between deviates from the same underlying Dirichlet distribution when the
#'   categories in each distribution are not allowed to be reordered.
#'
#' @param x A list such that each element is a 2D double vector with
#'   observations as the rows and each category within each observation as the
#'   columns
#' @param joint_alpha A vector of concentration parameters representing all
#'   raters
#' @param pairwise_alpha A list of lists of vectors of concentration parameters
#'   representing all pairs of raters
#' @param pairwise A boolean value indicating whether pairwise differences
#'   between individual raters should be computed; if \code{FALSE}, pairwise
#'   differences will be ignored in order to reduce the time and memory
#'   complexity of the computation
#' @param samples An integer value indicating how many samples to use to
#'   estimate expected differences
#' @param sampling_method A string indicating whether bootstrapped sampling
#'   should be performed using a \code{"parametric"} or \code{"nonparametric"}
#'   approach
#' @param verbose A boolean value indicating whether to provide progress updates
#' @return A two-element list indicating the expected differences between
#'   observations for all raters and for each pair of raters, respectively
#' @export
#' compute_single_expectation

compute_single_expectation <- function(x, joint_alpha, pairwise_alpha, pairwise, samples=1000, sampling_method="nonparametric", verbose=FALSE) {
   if(sampling_method=="nonparametric") { #Use nonparametric bootstrapping to estimate the expected differences
      de_individual_joint <- rep(0, nrow(x[[1]]))
      de_individual_pairwise <- vector("list",(length(x)-1))
      for(i in 1:length(de_individual_pairwise)) {
         de_individual_pairwise[[i]] <- vector("list",length(x))
         for(j in (i+1):length(de_individual_pairwise[[i]])) {
            de_individual_pairwise[[i]][[j]] <- rep(0, nrow(x[[1]]))
         }
      }

      for(i in 1:samples) {
         #Generate artificial data
         if(verbose) {
            print(paste0("Bootstrapping data: Creating sample ", i))
            utils::flush.console()
         }
         sample_data <- generate_artificial_data_nonparametric(x, shuffle=FALSE, shuffle_dimension=NA)

         #Compute expected differences
         differences <- compute_observed_differences(sample_data)
         de_individual_joint <- de_individual_joint + differences[[1]]
         for(i in 1:length(de_individual_pairwise)) {
            for(j in (i+1):length(de_individual_pairwise[[i]])) {
               de_individual_pairwise[[i]][[j]] <- de_individual_pairwise[[i]][[j]] + differences[[2]][[i]][[j]]
            }
         }
      }
      de_individual_joint <- Reduce(`+`, de_individual_joint) / samples
      for(i in 1:length(de_individual_pairwise)) {
         for(j in (i+1):length(de_individual_pairwise[[i]])) {
            de_individual_pairwise[[i]][[j]] <- Reduce(`+`, de_individual_pairwise[[i]][[j]]) / samples
         }
      }
   } else if(sampling_method=="parametric") { #Use the already-computed concentration parameters to exactly compute the expected differences
      de_individual_joint <- vector("double")
      sum_de_individual_joint <- vector("double")
      if(pairwise) {
         de_individual_pairwise <- vector("list",(length(pairwise_alpha)))
         for(i in 1:length(de_individual_pairwise)) {
            de_individual_pairwise[[i]] <- vector("list",length(pairwise_alpha[[i]]))
            for(j in (i+1):length(de_individual_pairwise[[i]])) {
               de_individual_pairwise[[i]][[j]] <- vector("double")
            }
         }
         sum_de_individual_pairwise <- de_individual_pairwise
      }

      for(i in 1:length(joint_alpha)) { #Compute D_e_joint for each distribution
         if(verbose) {
            print(paste0("Computing expected joint differences: Rater ", i))
            utils::flush.console()
         }
         de_individual_joint[i] <- 0
         sum_de_individual_joint[i] <- 0
         for(j in 1:length(joint_alpha[[i]])) { #Compute a_0
            sum_de_individual_joint[i] <- sum_de_individual_joint[i] + joint_alpha[[i]][j]
         }
         for(j in 1:length(joint_alpha[[i]])) { #Compute summation for D_e_joint
            de_individual_joint[i] <- de_individual_joint[i] +
                                      (((2*joint_alpha[[i]][j])*(sum_de_individual_joint[i]-joint_alpha[[i]][j])) /
                                      ((sum_de_individual_joint[i]^2)*(sum_de_individual_joint[i]+1)))
         }
      }
      de_individual_joint <- de_individual_joint / length(joint_alpha)

      if(pairwise) {
         for(i in 1:(length(pairwise_alpha[[1]])-1)) { #For each iteration i...
            for(j in (i+1):length(pairwise_alpha[[i]])) { #And for each fellow iteration j...
               if(verbose) {
                  print(paste0("Computing expected pairwise differences: Raters ", i, " and ", j))
                  utils::flush.console()
               }
               for(k in 1:nrow(pairwise_alpha[[i]][[j]])) { #Compute D_e_pairwise for each distribution
                  de_individual_pairwise[[i]][[j]][k] <- 0
                  sum_de_individual_pairwise[[i]][[j]][k] <- 0
                  for(l in 1:length(pairwise_alpha[[i]][[j]][k,])) { #Compute a_0
                     sum_de_individual_pairwise[[i]][[j]][k] <- sum_de_individual_pairwise[[i]][[j]][k] + pairwise_alpha[[i]][[j]][k,l]
                  }
                  for(l in 1:length(pairwise_alpha[[i]][[j]][k,])) { #Compute summation for D_e_pairwise
                     de_individual_pairwise[[i]][[j]][k] <- de_individual_pairwise[[i]][[j]][k] +
                                               (((2*pairwise_alpha[[i]][[j]][k,l])*(sum_de_individual_pairwise[[i]][[j]][k]-pairwise_alpha[[i]][[j]][k,l])) /
                                               ((sum_de_individual_pairwise[[i]][[j]][k]^2)*(sum_de_individual_pairwise[[i]][[j]][k]+1)))
                  }
               }
            }
            de_individual_pairwise[[i]][[j]] <- de_individual_pairwise[[i]][[j]] / length(nrow(pairwise_alpha[[i]][[j]]))
         }
      } else {
         de_individual_pairwise <- NA
      }
   }
   return(list(de_individual_joint, de_individual_pairwise))
}

#' Estimate Alpha from Data
#'
#' @description This helper function estimates the expected concentration
#'   parameters of the Dirichlet distribution underlying multiple observations.
#'
#' @param x A list such that each element is a 2D double vector with
#'   observations as the rows and each category within each observation as the
#'   columns
#' @param pairwise A boolean value indicating whether pairwise differences
#'   between individual raters should be computed; if \code{FALSE}, pairwise
#'   differences will be ignored in order to reduce the time and memory
#'   complexity of the computation
#' @param estimate_pairwise_alpha_from_joint A boolean value indicating, when
#'   estimating reliability for pairs of raters, whether each pair of raters
#'   should use the concentration parameters estimated across all raters
#'   (\code{TRUE}) or whether separate sets of concentration parameters should
#'   be estimated for each pair of raters (\code{FALSE}); this argument only has
#'   an effect when \code{alpha==NA}, and it is strongly suggested that the
#'   default value of \code{TRUE} be used for this argument when possible
#' @param symmetric_alpha A boolean value indicating whether concentration
#'   parameters should be assumed to be unchanged between observations, which is
#'   common in many topic modeling procedures
#' @param convcrit A numeric value indicating the threshold for convergence used
#'   to estimate concentration parameters
#' @param maxit A numeric value indicating the maximum number of iterations used
#'   to estimate concentration parameters
#' @param progress A boolean value indicating whether to provide progress
#'   updates
#' @return A two-element list indicating the estimated concentration parameters
#'   for all raters and for each pair of raters, respectively
#' @export
#' estimate_alpha_from_data

estimate_alpha_from_data <- function(x,pairwise,estimate_pairwise_alpha_from_joint, symmetric_alpha=FALSE, convcrit=0.00001, maxit=1000,
                                     progress=FALSE) {
   data <- vector("list",nrow(x[[1]]))
   joint_alpha <- data
   pairwise_alpha <- NA
   for(i in 1:nrow(x[[1]])) { #Extract a single distribution across multiple iterations
      if(progress) {
         print(paste0("Reorganizing data: Row ", i))
         utils::flush.console()
      }
      data[[i]] <- x[[1]][i,]
      for(j in 2:length(x)) {
         data[[i]] <- rbind(data[[i]], x[[j]][i,])
      }
   }
   #Create list with one element for each observation, each of which is a vector of alphas
   for(i in 1:length(data)) {
      if(progress) {
         print(paste0("Estimating omnibus concentration parameters: Rater ", i))
         utils::flush.console()
      }
      if(symmetric_alpha) { #Equally partition alpha0 among all categories
         generated_alphas <- symmetric.dirichlet.mle(data[[i]], convcrit=convcrit, maxit=maxit, progress=progress)
         joint_alpha[[i]] <- rep((generated_alphas$`alpha0`/ length(generated_alphas$`alpha`)),length(generated_alphas$`alpha`))
      } else { #Use alpha as-is, potentially with different values for each category; may be extremely slow for distributions with many categories
         joint_alpha[[i]] <- dirichlet.mle(data[[i]], convcrit=convcrit, maxit=maxit, progress=progress)$`alpha`
      }
   }
   #Create nested list: top level is iterations, second level is iterations,
   #third level is distributions, each of which is a vector of alphas
   if(pairwise) {
      pairwise_alpha <- vector("list",(length(x)-1))
      pairwise_alpha[1:length(x)] <- list(vector("list",length(x)))
      if(estimate_pairwise_alpha_from_joint) {
         alphas_to_fill_in <- do.call("rbind", joint_alpha)
         for(i in 1:(length(x)-1)) {
            if(progress) {
               print(paste0("Estimating pairwise concentration parameters: Rater ", i))
               utils::flush.console()
            }
            pairwise_alpha[[i]][(i+1):length(x)] <- list(alphas_to_fill_in)
         }
      } else { #Warning: May be extremely slow; setting estimate_pairwise_alpha_from_joint=TRUE is strongly advised
         for(i in 1:(length(x)-1)) {
            for(j in (i+1):length(x)) {
               if(progress) {
                  print(paste0("Estimating pairwise concentration parameters: Raters ", i, " and ", j))
                  utils::flush.console()
               }
               estimated_alphas = vector("list",nrow(x[[i]]))
               for(k in 1:nrow(x[[i]])) {
                  if(symmetric_alpha) { #Equally partition alpha0 among all categories
                      generated_alphas <- symmetric.dirichlet.mle(rbind(x[[i]][k,],x[[j]][k,]), convcrit=convcrit, maxit=maxit, progress=progress)
                      estimated_alphas[[k]] <- rep((generated_alphas$`alpha0`/length(generated_alphas$`alpha`)),
                                                            length(generated_alphas$`alpha`))
                  } else {
                      estimated_alphas[[k]] <- dirichlet.mle(rbind(x[[i]][k,],x[[j]][k,]), convcrit=convcrit, maxit=maxit, progress=progress)$`alpha`
                  }
               }
               pairwise_alpha[[i]][[j]] <- do.call("rbind", estimated_alphas)
            }
         }
      }
   } else {
      pairwise_alpha <- NA
   }
   return(list(joint_alpha,pairwise_alpha))
}

#' Distance Between Clusters
#'
#' @description This helper function computes the distance between two clusters
#'   based on a specified metric.
#'
#' @param x_obs A vector of indices representing the observations in the first
#'   cluster
#' @param y_obs A vector of indices representing the observations in the second
#'   cluster
#' @param alldiff A 2D array indicating the distance between each pair of
#'   observations across all raters
#' @param x_restructured A 2D array containing the original data set, restructured to
#'   facilitate more straightforward references in this function
#' @param clustering_method A string indicating the criterion that will be used
#'   to evaluate the distance between clusters; permitted values are
#'   \code{"average"}, \code{"minimum"}, \code{"maximum"}, \code{"median"},
#'   \code{"centroid"}, and \code{"wald"}
#' @return A numeric value indicating the distance between clusters
#' @export
#' clustering_determine_difference

clustering_determine_difference <- function(x_obs, y_obs, alldiff=NA, x_restructured=NA, clustering_method=NA) {
   #Compute distance between clusters
   if(clustering_method %in% c("average", "maximum", "minimum", "median")) {
      currentdiff <- alldiff[x_obs, y_obs]
   }
   if(clustering_method == "average") {
      thisdiff <- mean(currentdiff)
   } else if(clustering_method == "maximum") {
      thisdiff <- max(currentdiff)
   } else if(clustering_method == "minimum") {
      thisdiff <- min(currentdiff)
   } else if(clustering_method == "median") {
      thisdiff <- stats::median(currentdiff)
   } else if(clustering_method == "centroid") {
      if(length(x_obs) > 1) x_mu <- colMeans(x_restructured[x_obs,]) else x_mu <- x_restructured[x_obs,]
      if(length(y_obs) > 1) y_mu <- colMeans(x_restructured[y_obs,]) else y_mu <- x_restructured[y_obs,]
      thisdiff <- squarediff(x_mu, y_mu)
   } else if(clustering_method == "ward") {
      if(length(x_obs) > 1) x_mu <- colMeans(x_restructured[x_obs,]) else x_mu <- x_restructured[x_obs,]
      if(length(y_obs) > 1) y_mu <- colMeans(x_restructured[y_obs,]) else y_mu <- x_restructured[y_obs,]
      xy_mu <- colMeans(x_restructured[c(x_obs,y_obs),])
      if(length(x_obs) > 1) ssx <- sum(sweep(x_restructured[x_obs,], 2, x_mu)^2) #Compute current sum of squares for x_obs else ssx <- 0
      if(length(y_obs) > 1) ssy <- sum(sweep(x_restructured[y_obs,], 2, y_mu)^2) #Compute current sum of squares for y_obs else ssy <- 0
      ssxy <- sum(sweep(x_restructured[c(x_obs,y_obs),], 2, xy_mu)^2) #Compute sum of squares for a potential cluster comprising x_obs+y_obs
      thisdiff <- ssxy - ssx - ssy #We want to select the merge that minimizes the within-cluster sum of squares increase
   } else {
      stop("Invalid value set for clustering_method.")
   }
   return(thisdiff)
}

#' Sum of Squared Differences
#'
#' @description This helper function computes the sum of squared differences
#'   between two vectors.
#'
#' @param x A numeric vector
#' @param y A numeric vector
#' @return A numeric value representing the sum of the squared differences
#'   between \code{x} and \code{y}
#' @export
#' squarediff

squarediff <- function(x,y) sum((x-y)^2)

#' Restricted Divisive Clustering
#'
#' @description This helper function performs divisive hierarchical cluster
#'   analysis in order to minimize the sum of squared differences between
#'   raters.
#'
#' @details This procedure conducts a cluster analysis in order to group topics
#'   from each rater into clusters indicating optimal matches between those
#'   topics. This effectively serves to reorder the topics constructed by each
#'   such that, to the extent possible, each individual topic is matched with
#'   its best-fitting counterparts constructed by each other rater before
#'   reliability is assessed.
#'
#'   This cluster analysis uses an additional constraint: all clusters resulting
#'   from dividing an existing cluster must contain a number of observations
#'   that is a multiple of the number of topics constructed by each rater, and
#'   all raters must be equally represented in each of those resulting clusters.
#'   This ensures that at the end of the process, the number of clusters will be
#'   equal to the number of topics constructed by each rater, and each rater
#'   will have yielded one topic belonging to each cluster.
#'
#'   However, this approach sometimes creates clusters based on macro-level
#'   dynamics rather than micro-level differences between topics, ultimately
#'   yielding weakly matched sets of categories compared with other approaches.
#'   In those cases, either \link{rotational_shuffling} or
#'   \link{restricted_agglomerative_clustering} may yield superior results.
#'
#' @param x A list such that each element is a 2D double vector, with the topics
#'   to be reordered representing the rows of each vector
#' @param verbose A boolean value indicating whether to provide progress updates
#' @param clustering_method A string indicating the criterion that will be used
#'   to merge or divide clusters; permitted values are \code{"average"},
#'   \code{"minimum"}, \code{"maximum"}, \code{"median"}, \code{"centroid"}, and
#'   \code{"wald"}
#' @param slow_clustering A boolean value indicating whether the distances
#'   between clusters should be recalculated after each individual observation
#'   is added to the new cluster (\code{TRUE}), which can improve the
#'   cohesiveness of the resulting cluster, or whether all observations should
#'   be added to the new cluster without recomputing the distances between
#'   clusters after every addition (\code{FALSE})
#' @return A two-element list containing the reordered data set and a matrix
#'   indicating the relationship between each reordered topic and its original
#'   position in the data, respectively
#' @export
#' restricted_divisive_clustering

restricted_divisive_clustering <- function(x, verbose=FALSE, clustering_method="average", slow_clustering=FALSE) {

   #Performs divisive cluster analysis on the categories obtained across all cross-validation
   #iterations.
   #
   #All cluster divisions are performed in blocks of k, with all k topics equally represented in any
   #any given cluster. This ensures that at the end of the process, we have k clusters such that each
   #cross-validation iteration yielded one category belonging to each cluster.
   #
   #However, this approach tends to create clusters based on macro-level dynamics rather than
   #micro-level differences. This tends to yield weakly matched sets of categories compared with
   #other approaches.

   target_number_of_clusters <- nrow(x[[1]])
   c <- length(x) #Number of iterations
   k <- nrow(x[[1]]) #Number of categories (e.g., topics)
   assignments <- matrix(1, k, c)
   current_number_of_clusters <- 1
   current_cluster_labels <- as.numeric(labels(table(assignments))$assignments)

   if(verbose) {
      print(paste0("Computing preliminary difference matrix"))
      utils::flush.console()
   }

   alldiff <- array(data = NA, dim = c(c*k,c*k))
   if(clustering_method %in% c("average", "minimum", "maximum", "median")) {
      alldiff[,] <- sapply(1:c, function(q) sapply(1:k, function(s) sapply(1:c, function(p) sapply(1:k, function(r) squarediff(x[[p]][r,], x[[q]][s,])))), simplify="array")
   } else if(clustering_method=="centroid" | clustering_method=="wald") {
      x_restructured <- array(data = NA, dim = c(c*k,length(x[[1]][1,]))) #Create a new version of the raw allocations that is easier to query on-the-fly
      for(i in 1:(c*k)) {
         x_restructured[i,] <- x[[ceiling(i/k)]][if((i %% k) == 0) k else (i %% k),]
      }
   }

   current_number_of_clusters <- 1
   while(current_number_of_clusters < target_number_of_clusters) {
      if(verbose) {
         print(paste0("Restricted divisive clustering: Beginning step ", current_number_of_clusters))
         utils::flush.console()
      }
      current_maximum_outlier <- 0
      current_maximum_outlier_value <- -Inf

      #First, identify the observation that is most dissimilar from its own cluster, and create a new cluster with it
      for(i in 1:length(assignments)) {
         x_obs <- i
         y_obs <- which(assignments == assignments[i])
         if(length(y_obs) <= c) { #This cluster has already reached its minimum size and cannot be split further
            next
         } else { #This cluster can still be split further
            thisdiff <- clustering_determine_difference(x_obs, y_obs, alldiff, x_restructured, clustering_method)
         }
         if(thisdiff > current_maximum_outlier_value) {
            current_maximum_outlier <- i
            current_maximum_outlier_value <- thisdiff
         }
      }
      old_cluster_label <- assignments[current_maximum_outlier]
      current_number_of_clusters <- current_number_of_clusters + 1
      assignments[current_maximum_outlier] <- current_number_of_clusters
      column <- ceiling(current_maximum_outlier/k)

      #Second, identify the k-1 observations (from the other k-1 categories) that are as close as possible to the new cluster and as far as possible from the old cluster, and reclassify each successive observation identified to the new cluster
      categories_represented_in_new_cluster <- rep(FALSE, c)
      categories_represented_in_new_cluster[column] <- TRUE
      while(sum(categories_represented_in_new_cluster) < c) {
         current_maximum_outlier <- rep(0, c)
         current_maximum_outlier_value <- rep(-Inf, c)
         for(i in which(assignments == old_cluster_label)) {
            column <- ceiling(i/k)
            if(categories_represented_in_new_cluster[column]) { #If we've already reclassified an observation from this column into the new cluster, then we don't need to reclassify another one
               next
            }
            x_obs <- i
            y_obs <- which(assignments == current_number_of_clusters) #The new cluster
            z_obs <- which(assignments == old_cluster_label) #The old cluster
            newdiff <- clustering_determine_difference(x_obs, y_obs, alldiff, x_restructured, clustering_method)
            olddiff <- clustering_determine_difference(x_obs, z_obs, alldiff, x_restructured, clustering_method)
            if(olddiff - newdiff > current_maximum_outlier_value[column]) { #This observation is an improvement over the current one attributed to this category
               current_maximum_outlier[column] <- i
               current_maximum_outlier_value[column] <- olddiff - newdiff
            }
         }
         if(slow_clustering) {
            best_outlier <- max(current_maximum_outlier_value)
            current_maximum_outlier <- current_maximum_outlier[which(current_maximum_outlier_value==max(current_maximum_outlier_value))][1]
         } else {
            current_maximum_outlier <- current_maximum_outlier[current_maximum_outlier > 0]
         }
         assignments[current_maximum_outlier] <- current_number_of_clusters
         categories_represented_in_new_cluster[ceiling(current_maximum_outlier/k)] <- TRUE
      }

      #Third, identify another candidate set of k observations and, if they are collectively closer to the new closer than the old cluster, reclassify them as well; repeat until no further set of k observations can be reclassified to the new cluster
      while((sum(categories_represented_in_new_cluster) < c) && (length(which(assignments == old_cluster_label)) > k)) {
         temp_assignments <- assignments
         candidate_observations <- numeric(0)
         categories_represented_in_new_cluster <- rep(FALSE, c)
         #Begin by identifying a candidate set of k observations
         while(sum(categories_represented_in_new_cluster) < c) {
            current_maximum_outlier <- rep(0, c)
            current_maximum_outlier_value <- rep(-Inf, c)
            for(i in which(temp_assignments == old_cluster_label)) {
               column <- ceiling(i/k)
               if(categories_represented_in_new_cluster[column]) { #If we've already reclassified an observation from this column into the new cluster, then we don't need to reclassify another one
                  next
               }
               x_obs <- i
               y_obs <- which(temp_assignments == current_number_of_clusters) #The new cluster
               z_obs <- which(temp_assignments == old_cluster_label) #The old cluster
               newdiff <- clustering_determine_difference(x_obs, y_obs, alldiff, x_restructured, clustering_method)
               olddiff <- clustering_determine_difference(x_obs, z_obs, alldiff, x_restructured, clustering_method)
               if(olddiff - newdiff > current_maximum_outlier_value[column]) { #This observation is an improvement over the current one attributed to this category
                  current_maximum_outlier[column] <- i
                  current_maximum_outlier_value[column] <- olddiff - newdiff
               }
            }
            if(slow_clustering) {
               best_outlier <- max(current_maximum_outlier_value)
               current_maximum_outlier <- current_maximum_outlier[which(current_maximum_outlier_value==max(current_maximum_outlier_value))][1]
            } else {
               current_maximum_outlier <- current_maximum_outlier[current_maximum_outlier > 0]
            }
            temp_assignments[current_maximum_outlier] <- current_number_of_clusters
            categories_represented_in_new_cluster[ceiling(current_maximum_outlier/k)] <- TRUE
            candidate_observations <- c(candidate_observations, current_maximum_outlier)
         }
         #Then, see if this set of observations is closer to the new cluster than the old one, reclassify it to the new cluster; if not, then break this inner loop and start the next cluster division
         x_obs <- candidate_observations
         y_obs <- which(assignments == current_number_of_clusters) #The new cluster
         z_obs <- which(temp_assignments == old_cluster_label) #The old cluster
         newdiff <- clustering_determine_difference(x_obs, y_obs, alldiff, x_restructured, clustering_method)
         olddiff <- clustering_determine_difference(x_obs, z_obs, alldiff, x_restructured, clustering_method)
         if(olddiff > newdiff) {
            assignments[candidate_observations] <- current_number_of_clusters
         } else { #Reclassifying these observations to the new cluster would make the cluster classifications worse, not better
            break
         }
      }
   }
   #Reorganize the assignments matrix so that each row will show the order of topics
   new_assignments <- assignments * 0
   for(i in 1:nrow(assignments)) {
      for(j in 1:ncol(assignments)) {
         new_assignments[i,j] <- which(assignments[,j]==i)
      }
   }
   
   return(t(new_assignments))
}

#' Restricted Agglomerative Clustering
#'
#' @description This helper function performs agglomerative hierarchical cluster
#'   analysis in order to minimize the sum of squared differences between
#'   raters.
#'
#' @details This procedure conducts a cluster analysis in order to group topics
#'   from each rater into clusters indicating optimal matches between those
#'   topics. This effectively serves to reorder the topics constructed by each
#'   such that, to the extent possible, each individual topic is matched with
#'   its best-fitting counterparts constructed by each other rater before
#'   reliability is assessed.
#'
#'   This cluster analysis uses an additional constraint: topics from the same
#'   rater can never be combined into the same cluster. This ensures that at the
#'   end of the process, the number of clusters will be equal to the number of
#'   topics constructed by each rater, and each rater will have yielded one
#'   topic belonging to each cluster. If two clusters would be combined such
#'   that this rule would be violated, then immediately afterward, all offending
#'   topics are removed from the newly merged cluster and each one is treated as
#'   an independent cluster to be later merged into other clusters.
#'
#'   Additionally, to prevent infinite loops (e.g., the same topic is repeatedly
#'   added to a larger cluster and removed afterward in accordance with the
#'   preceding procedure), two additional requirements are imposed in order to
#'   merge any pair of clusters. First, when considering any potential cluster
#'   merge operation, if executing the merge would in an identical set of
#'   clusters as any prior step, that operation is not performed, and the
#'   procedure skips to the next pair of clusters to be considered for merging.
#'   Second, when considering any potential cluster merge operation, the
#'   procedure compares the set of raters from which all topics in each of the
#'   two clusters were drawn, and if either one is a perfect subset of the other
#'   (e.g., the topics in cluster 1 came from raters 3 and 6, while the topics
#'   in cluster 2 were created by raters 1, 2, 3, 4, 6, and 8), then we cannot
#'   merge the two clusters, as all observations in one cluster would just be
#'   removed from the resulting cluster afterward. Whenever such a perfect
#'   subset is observed via this diagnostic, the procedure skips to the next
#'   pair of clusters to be considered for merging.
#'
#'   Notably, although infinite loops are avoided, this function still tends to
#'   get stuck in long-lasting loops. When this happens, either
#'   \link{rotational_shuffling} or \link{restricted_divisive_clustering} may be
#'   a better choice.
#'
#' @param x A list such that each element is a 2D double vector, with the topics
#'   to be reordered representing the rows of each vector
#' @param verbose A boolean value indicating whether to provide progress updates
#' @param clustering_method A string indicating the criterion that will be used
#'   to merge or divide clusters; permitted values are \code{"average"},
#'   \code{"minimum"}, \code{"maximum"}, \code{"median"}, \code{"centroid"}, and
#'   \code{"wald"}
#' @return A two-element list containing the reordered data set and a matrix
#'   indicating the relationship between each reordered topic and its original
#'   position in the data, respectively
#' @export
#' restricted_agglomerative_clustering

restricted_agglomerative_clustering <- function(x, verbose=FALSE, clustering_method="average") {

   target_number_of_clusters <- nrow(x[[1]])
   c <- length(x) #Number of iterations
   k <- nrow(x[[1]]) #Number of categories (e.g., topics)
   assignments <- matrix(1:(k*c), k, c)
   current_number_of_clusters <- length(table(assignments))
   current_cluster_labels <- as.numeric(labels(table(assignments))$assignments)

   if(verbose) {
      print(paste0("Computing preliminary difference matrix"))
      utils::flush.console()
   }

   alldiff <- array(data = NA, dim = c(c*k,c*k))
   if(clustering_method %in% c("average", "minimum", "maximum", "median")) {
      alldiff[,] <- sapply(1:c, function(q) sapply(1:k, function(s) sapply(1:c, function(p) sapply(1:k, function(r) squarediff(x[[p]][r,], x[[q]][s,])))), simplify="array")
   } else if(clustering_method=="centroid" | clustering_method=="wald") {
      x_restructured <- array(data = NA, dim = c(c*k,length(x[[1]][1,]))) #Create a new version of the raw allocations that is easier to query on-the-fly
      for(i in 1:(c*k)) {
         x_restructured[i,] <- x[[ceiling(i/k)]][if((i %% k) == 0) k else (i %% k),]
      }
   }

   step <- 0
   past_assignments <- vector("list")
   past_assignments_number_of_clusters <- vector("numeric")
   while(current_number_of_clusters > target_number_of_clusters) {
      step <- step + 1
      if(verbose) {
         print(paste0("Restricted agglomerative clustering: Beginning step ", step))
         utils::flush.console()
      }
      current_best_merge <- c(0,0)
      current_best_merge_value <- Inf
      for(i in 1:(current_number_of_clusters-1)) {
         x_obs <- which(assignments == current_cluster_labels[i]) #Retrieve all indices in assignments whose value matches current_cluster_labels[i]
         for(j in (i+1):current_number_of_clusters) {
            y_obs <- which(assignments == current_cluster_labels[j]) #Retrieve all indices in assignments whose value matches current_cluster_labels[j]

            #Subroutine to check for perfect "subset" between clusters based on the cross-validation iterations from which they were drawn
            x_obs_cv <- ceiling(x_obs / k)
            y_obs_cv <- ceiling(y_obs / k)
            if(length(unique(c(x_obs_cv, y_obs_cv))) <= max(length(x_obs_cv), length(y_obs_cv))) {
               next
            }

            #Compute distance between clusters
            if(clustering_method %in% c("average", "maximum", "minimum", "median")) {
               currentdiff <- alldiff[x_obs, y_obs]
            }
            if(clustering_method == "average") {
               thisdiff <- mean(currentdiff)
            } else if(clustering_method == "maximum") {
               thisdiff <- max(currentdiff)
            } else if(clustering_method == "minimum") {
               thisdiff <- min(currentdiff)
            } else if(clustering_method == "median") {
               thisdiff <- stats::median(currentdiff)
            } else if(clustering_method == "centroid") {
               if(length(x_obs) > 1) x_mu <- colMeans(x_restructured[x_obs,]) else x_mu <- x_restructured[x_obs,]
               if(length(y_obs) > 1) y_mu <- colMeans(x_restructured[y_obs,]) else y_mu <- x_restructured[y_obs,]
               thisdiff <- squarediff(x_mu, y_mu)
            } else if(clustering_method == "ward") {
               if(length(x_obs) > 1) x_mu <- colMeans(x_restructured[x_obs,]) else x_mu <- x_restructured[x_obs,]
               if(length(y_obs) > 1) y_mu <- colMeans(x_restructured[y_obs,]) else y_mu <- x_restructured[y_obs,]
               xy_mu <- colMeans(x_restructured[c(x_obs,y_obs),])
               if(length(x_obs) > 1) ssx <- sum(sweep(x_restructured[x_obs,], 2, x_mu)^2) #Compute current sum of squares for x_obs else ssx <- 0
               if(length(y_obs) > 1) ssy <- sum(sweep(x_restructured[y_obs,], 2, y_mu)^2) #Compute current sum of squares for y_obs else ssy <- 0
               ssxy <- sum(sweep(x_restructured[c(x_obs,y_obs),], 2, xy_mu)^2) #Compute sum of squares for a potential cluster comprising x_obs+y_obs
               thisdiff <- ssxy - ssx - ssy #We want to select the merge that minimizes the within-cluster sum of squares increase
            } else {
               stop("Invalid value set for clustering_method.")
            }

            if(thisdiff < current_best_merge_value) {
               #If the resulting assignments would exactly duplicate a past set of assignments, then we are stuck in an infinite loop, so we should not perform this merge--we need to do something else instead
               temp_assignments <- assignments
               temp_current_number_of_clusters <- length(table(temp_assignments))
               y_obs <- which(temp_assignments == current_cluster_labels[j])
               temp_assignments[y_obs] <- current_cluster_labels[i]
               valid_merge <- TRUE
               if(step > 1) {
                  past_assignments_with_same_number_of_clusters <- which(past_assignments_number_of_clusters == temp_current_number_of_clusters)
                  if(length(past_assignments_with_same_number_of_clusters) > 0) {
                     for(l in 1:length(past_assignments_with_same_number_of_clusters)) {
                        if(all.equal(temp_assignments, past_assignments[[past_assignments_with_same_number_of_clusters[l]]]) == TRUE) { #We've seen this list of assignments in the past and therefore should not perform this particular merge; a different merge should be implemented instead
                           valid_merge = FALSE
                           break
                        }
                     }
                  }
               }
               if(valid_merge) {
                  current_best_merge <- c(current_cluster_labels[i], current_cluster_labels[j])
                  current_best_merge_value <- thisdiff
               }
            }
         }
      }

      #For the closest pair of clusters, set the same assignment to all categories they contain, thereby merging the clusters
      y_obs <- which(assignments == current_best_merge[2])
      assignments[y_obs] <- current_best_merge[1]

      #Prepare to add the new assignments to the list of past assignments
      temp_assignments <- assignments
      temp_current_number_of_clusters <- length(table(temp_assignments))
      temp_current_cluster_labels <- as.numeric(labels(table(temp_assignments))$temp_assignments)
      for(i in 1:temp_current_number_of_clusters) {
         thislabel <- which(temp_assignments == temp_current_cluster_labels[i])
         temp_assignments[thislabel] <- i
      }

      #Add the new assignments to the list of past assignments
      past_assignments[[step]] <- temp_assignments
      past_assignments_number_of_clusters[step] <- temp_current_number_of_clusters

      #If the last merge operation resulted in any cross-validation iteration having two categories
      #assigned to the merged cluster--represented as redundant values in the same column of assignments--
      #then remove those observations from the cluster and make them their own individual clusters again,
      #such that they will ultimately be re-merged into different clusters later
      for(i in 1:c) {
         dupassign <- which(assignments[,i] == current_best_merge[1])
         if(length(dupassign) > 1) {
            current_maximum_cluster <- max(current_cluster_labels)
            assignments[dupassign,i] <- (current_maximum_cluster+1):(current_maximum_cluster+length(dupassign))
         }
      }

      current_number_of_clusters <- length(table(assignments))
      current_cluster_labels <- as.numeric(labels(table(assignments))$assignments)

      #Reorder the cluster labels based on the index of their lowest element
      max_assignments <- max(assignments)
      assignments <- assignments + max_assignments
      current_maximum <- 1
      for(i in 1:length(assignments)) {
         if(assignments[i] > max_assignments) {
            current_cluster <- which(assignments == assignments[i])
            assignments[current_cluster] <- current_maximum
            current_maximum <- current_maximum+1
         }
      }
   }

   #Relabel the clusters from 1:k
   for(i in 1:k) {
      thislabel <- which(assignments == current_cluster_labels[i])
      assignments[thislabel] <- i
   }

   #Reorganize the assignments matrix so that each row will show the order of topics
   new_assignments <- assignments * 0
   for(i in 1:nrow(assignments)) {
      for(j in 1:ncol(assignments)) {
         new_assignments[i,j] <- which(assignments[,j]==i)
      }
   }
   
   return(t(new_assignments))
}

#' Rotational Shuffling
#'
#' @description This helper function reorders the rows of ratings from multiple
#'   raters in order to minimize the sum of squared differences between them.
#'
#' @param x A list such that each element is a 2D double vector, with the topics
#'   to be reordered representing the rows of each vector
#' @param verbose A boolean value indicating whether to provide progress updates
#' @param swap_tol A numeric value representing the minimum threshold by which
#'   the sum of squared differences must be improved in order to execute a
#'   potential swap of two rows; this is necessary in order to prevent infinite
#'   loops due to minuscule rounding errors
#' @return A two-element list containing the reordered data set and a matrix
#'   indicating the relationship between each reordered topic and its original
#'   position in the data, respectively
#' @export
#' rotational_shuffling

rotational_shuffling <- function(x, verbose=FALSE, swap_tol=(10^(-12))) {

   #Minimizes differences between the categories obtained across all cross-validation iterations by
   #repeatedly swapping matched pairs in order to reduce those differences. This approach converges
   #to a local minimum. However, it can be slow in some cases. If the speed of convergence is
   #insufficient, the agglomerative or divisive clustering approach may be appropriate alternatives.

   step <- 0 #Used to track progress when verbose=TRUE
   c <- length(x) #Number of iterations
   k <- nrow(x[[1]]) #Number of distributions (e.g., topics)
   v <- ncol(x[[1]]) #Number of categories in each distribution (e.g., words)
   assignments <- t(matrix(1:k, k, c))

   if(verbose) {
      print(paste0("Computing preliminary difference matrix"))
      utils::flush.console()
   }

   alldiff <- array(data = NA, dim = c(c,c,k,k))
   for(p in 1:(c-1)) {
      for(q in (p+1):c) {
         for(r in 1:k) {
             alldiff[p,q,r,1:k] <- rowSums(sweep(x[[q]], 2, x[[p]][r,])^2)
         }
      }
   }

   #Reduce d_o by repeatedly switching two distributions in one iteration
   repeat {
      if(verbose) {
         step <- step + 1
         print(paste0("Distribution switching: Beginning step ", step))
         utils::flush.console()
      }
      improvement <- 0
      iter <- 0
      dist1 <- 0
      dist2 <- 0

      for(p in 1:c) {
         for(r in 1:(k-1)) {
            for(s in (r+1):k) {
               change <- 0
               for(q in 1:c) {
                  if(p != q) {
                     iter1 <- min(p, q)
                     iter2 <- max(p, q)
                     change <- change + alldiff[iter1,iter2,assignments[iter1,r],assignments[iter2,r]] + alldiff[iter1,iter2,assignments[iter1,s],assignments[iter2,s]] - alldiff[iter1,iter2,assignments[iter1,r],assignments[iter2,s]] - alldiff[iter1,iter2,assignments[iter1,s],assignments[iter2,r]]
                  }
               }
               if(change > improvement) {
                  improvement <- change
                  iter <- p
                  dist1 <- r
                  dist2 <- s
               }
            }
         }
      }

      if(improvement > swap_tol) {
         temp_int = assignments[iter,dist1]
         assignments[iter,dist1] = assignments[iter,dist2]
         assignments[iter,dist2] = temp_int         
      } else {
         break
      }
   }
   return(assignments)
}

#' Shuffle Distributions
#'
#' @description This helper function reorders the categories in multiple
#'   Dirichlet deviates in order to minimize the sum of squared differences
#'   between them.
#'
#' @param x A list such that each element is a 2D double vector with
#'   observations as the rows and each category within each observation as the
#'   columns
#' @param verbose A boolean value indicating whether to provide progress updates
#' @param shuffle_method A string indicating what method (\code{"rotational"},
#'   \code{"agglomerative"}, or \code{"divisive"}) will be used to reorder
#'   topics
#' @param clustering_method A string indicating the criterion that will be used
#'   to merge or divide clusters if \code{shuffle_method=="agglomerative"} or
#'   \code{shuffle_method=="divisive"}; permitted values are \code{"average"},
#'   \code{"minimum"}, \code{"maximum"}, \code{"median"}, \code{"centroid"}, and
#'   \code{"wald"}
#' @param slow_clustering A boolean value indicating whether the distances
#'   between clusters should be recalculated after each individual observation
#'   is added to the new cluster (\code{TRUE}), which can improve the
#'   cohesiveness of the resulting cluster, or whether all observations should
#'   be added to the new cluster without recomputing the distances between
#'   clusters after every addition (\code{FALSE}); this argument only has an
#'   effect if \code{shuffle_method=="divisive"}
#' @param shuffle_dimension A string indicating whether \code{"rows"} or
#'   \code{"columns"} should be reordered
#' @return A two-element list containing the reordered data set and a matrix
#'   indicating the relationship between each reordered topic and its original
#'   position in the data, respectively
#' @export
#' shuffle_distributions

shuffle_distributions <- function(x, verbose=FALSE, shuffle_method="rotational", clustering_method="average", slow_clustering=FALSE, shuffle_dimension="rows") {
   if(shuffle_dimension=="columns") { #If we need to shuffle the categories rather than the distributions, we can deal with
                                      #this by transposing each cross-validation iteration, then retransposing them afterward.
      for(i in 1:length(x)) {
         x[[i]] <- t(x[[i]])
      }
   }

   c <- length(x) #Number of iterations
   k <- nrow(x[[1]]) #Number of distributions (e.g., topics)
   v <- ncol(x[[1]]) #Number of categories in each distribution (e.g., words)

   if(shuffle_method=="agglomerative") {
      assignments <- restricted_agglomerative_clustering(x, verbose, clustering_method)
   } else if(shuffle_method=="divisive") {
      assignments <- restricted_divisive_clustering(x, verbose, clustering_method, slow_clustering=FALSE)
   } else if(shuffle_method=="rotational") { #Like a Rubik's cube, this approach swaps pairs of distributions within a given iteration, one at a time, until the overall fit can no longer be improved with any additional swap
      assignments <- rotational_shuffling(x, verbose)
   } else {
      stop("Invalid value set for shuffle_method.")
      return(NA)
   }

   #Reorder the distributions in accordance with the assignments matrix
   for(i in 1:c) {
      x[[i]] <- x[[i]][assignments[i,1:length(assignments[i,])],]
   }

   if(shuffle_dimension=="columns") { #Undo the transposition of each cross-validation iteration
      for(i in 1:length(x)) {
         x[[i]] <- t(x[[i]])
      }
   }

   return(list(x, assignments))
}

#' Sirt Digamma 1
#'
#' @description This helper function estimates the derivative of the digamma
#'   function. This function is directly drawn from
#'   \code{sirt::sirt_digamma1}, as published at
#'   \url{https://github.com/cran/sirt/blob/master/R/sirt_digamma1.R}. This
#'   function does not appear to be included in current releases of \code{sirt}
#'   and therefore cannot be directly loaded from that package.
#'
#' @param x A vector of concentration parameters
#' @param h A numeric value used to define a pair of nearby observations
#' @return A list of the estimated concentration parameters, the sum of those
#'   estimated concentration parameters, and the ratio between each estimated
#'   concentration parameter and the sum of those parameters
#' @export
#' sirt_digamma1

sirt_digamma1 <- function(x, h=1e-3) { #Drawn from https://rdrr.io/cran/sirt/src/R/sirt_digamma1.R
   (digamma(x+h) - digamma(x-h)) / (2*h)
}

#' Estimate Concentration Parameters
#'
#' @description This helper function estimates the concentration parameters of
#'   the Dirichlet distribution underlying multiple observations, with no
#'   restrictions placed on the values of those concentration parameters. This
#'   function is heavily based on \code{\link[sirt]{dirichlet.mle}}, with
#'   modifications to avoid potential singularities in the estimation procedure.
#'
#' @param x A list such that each element is a 2D double vector with
#'   observations as the rows and each category within each observation as the
#'   columns
#' @param weights A numeric vector used to calibrate the initial estimates of
#'   concentration parameters
#' @param eps A numeric value used as a tolerance parameter to prevent
#'   logarithms of zero
#' @param convcrit A numeric value indicating the threshold for convergence used
#'   to estimate concentration parameters
#' @param maxit A numeric value indicating the maximum number of iterations used
#'   to estimate concentration parameters
#' @param oldfac A numeric value between 0 and 1 used as the convergence
#'   acceleration factor
#' @param progress A boolean value indicating whether progress updates should be
#'   provided
#' @return A list of the estimated concentration parameters, the sum of those
#'   estimated concentration parameters, and the ratio between each estimated
#'   concentration parameter and the sum of those parameters
#' @export
#' dirichlet.mle

dirichlet.mle <- function( x, weights=NULL, eps=10^(-5), convcrit=.00001,
                           maxit=1000, oldfac=.3, progress=FALSE) {
   #Drawn from https://rdrr.io/cran/sirt/src/R/dirichlet.mle.R, but with minor modifications
   #(especially in the initial setting of xsi) to avoid potential singularities in the estimation
   #procedure.

   # compute log pbar
   x <- ( x+eps ) / ( 1 + 2*eps )
   x <- x / rowSums(x)
   N <- nrow(x)
   K <- ncol(x)
   if( is.null(weights) ){
      #weights <- rep(1,(N*K))
      weights <- rep(1, N)
   }
   weights <- N * weights / sum( weights )
   log.pbar <- colMeans( weights * log( x ) )
   # compute inits
   alphaprob <- colMeans( x * weights )
   p2 <- mean( unlist(x)^2 * weights )
   #p2 <- mean( x[,1]^2 * weights ) #Original version of this line, as the function appeared in the sirt package
   xsi <- ( mean(alphaprob) - p2 ) / ( p2 - ( mean(alphaprob) )^2 )
   #xsi <- ( alphaprob[1] - p2 ) / ( p2 - ( alphaprob[1] )^2 ) #Original version of this line, as the function appeared in the sirt package
   alpha <- xsi * alphaprob
   K1 <- matrix(1,K,K)
   conv <- 1
   iter <- 1
   alpha[ alpha < (1e-10) ] <- 1e-10 #This line was not in the original function as it appeared in the sirt package

   #--- BEGIN iterations
   while( ( conv > convcrit ) & (iter < maxit) ){
      if(progress) {
         print(paste0("Estimating concentration parameters: Beginning step ", iter))
         utils::flush.console()
      }
      alpha0 <- alpha
      g <- N * digamma( sum(alpha ) ) - N * digamma(alpha) + N * log.pbar
      z <- N * sirt_digamma1( sum(alpha ))
      H <- diag( -N * sirt_digamma1( alpha ) ) + z
      alpha <- alpha0 - solve(H, g )
      alpha[ alpha < (1e-10) ] <- 1e-10
      #alpha[ alpha < 0 ] <- 1e-10 #Original version of this line, as the function appeared in the sirt package
      alpha <- alpha0 + oldfac*( alpha - alpha0 )
      conv <- max( abs( alpha0 - alpha ) )
      iter <- iter+1
   }
   alpha0 <- sum(alpha)
   xsi <- alpha / alpha0
   res <- list( alpha=alpha, alpha0=alpha0, xsi=xsi )
   return(res)
}

#' Estimate Symmetric Concentration Parameters
#'
#' @description This helper function estimates the concentration parameters of
#'   the Dirichlet distribution underlying multiple deviates, assuming that
#'   those concentration parameters are all equal. This function is heavily
#'   based on \code{\link[sirt]{dirichlet.mle}}, with modifications to restrict
#'   all concentration parameters to be equal and to avoid potential
#'   singularities in the estimation procedure.
#'
#' @param x A list such that each element is a 2D double vector with
#'   observations as the rows and each category within each observation as the
#'   columns
#' @param weights A numeric vector used to calibrate the initial estimates of
#'   concentration parameters
#' @param eps A numeric value used as a tolerance parameter to prevent
#'   logarithms of zero
#' @param convcrit A numeric value indicating the threshold for convergence used
#'   to estimate concentration parameters
#' @param maxit A numeric value indicating the maximum number of iterations used
#'   to estimate concentration parameters
#' @param oldfac A numeric value between 0 and 1 used as the convergence
#'   acceleration factor
#' @param progress A boolean value indicating whether progress updates should be
#'   provided
#' @return A list of the estimated concentration parameters, the sum of those
#'   estimated concentration parameters, and the ratio between each estimated
#'   concentration parameter and the sum of those parameters
#' @export
#' symmetric.dirichlet.mle

symmetric.dirichlet.mle <- function( x, weights=NULL, eps=10^(-5), convcrit=.00001,
                                     maxit=1000, oldfac=.3, progress=FALSE) {
   #Drawn from https://rdrr.io/cran/sirt/src/R/dirichlet.mle.R, but with minor modifications
   #(especially in the initial setting of xsi) to avoid potential singularities in the estimation
   #procedure and to force all resulting concentration parameters to be identical.

   # compute log pbar
   x <- ( x+eps ) / ( 1 + 2*eps )
   x <- x / rowSums(x)
   N <- nrow(x)
   K <- ncol(x)
   if( is.null(weights) ){
      weights <- rep(1, N)
   }
   weights <- N * weights / sum( weights )
   log.pbar <- colMeans( weights * log( x ) )
   # compute inits
   alphaprob <- colMeans( x * weights )
   p2 <- mean( unlist(x)^2 * weights )
   #p2 <- mean( x[,1]^2 * weights ) #Original version of this line, as the function appeared in the sirt package
   xsi <- ( mean(alphaprob) - p2 ) / ( p2 - ( mean(alphaprob) )^2 )
   #xsi <- ( alphaprob[1] - p2 ) / ( p2 - ( alphaprob[1] )^2 ) #Original version of this line, as the function appeared in the sirt package
   alpha <- xsi * alphaprob
   K1 <- matrix(1,K,K)
   conv <- 1
   iter <- 1
   alpha[ alpha < (1e-10) ] <- 1e-10 #This line was not in the original function as it appeared in the sirt package
   alpha0 <- rep(mean(alpha),length(alpha))
   alpha <- alpha0

   #--- BEGIN iterations
   while( ( conv > convcrit ) & (iter < maxit) ){
      if(progress) {
         print(paste0("Estimating concentration parameters: Beginning step ", iter))
         utils::flush.console()
      }
      #Given a matrix defining of a system of equations, where the diagonal elements = a and the off-diagonal elements = b,
      #and the resulting vector is named g, the solution vector may be expressed as:
      #solution[i] = -g[i]*(a+(length(g)-2)*b)/((a+(length(g)-1)*b)*(b-a)) - g[i]*(b)/((a+(length(g)-1)*b)*(b-a)) + sum(g*b/((a+(length(g)-1)*b)*(b-a)))
      #In this case, the diagonal elements = (-N*K*(digamma(alpha+(1e-3))-digamma(alpha-(1e-3)))/(2*(1e-3)))[1]+z and the off-diagonal
      #elements = z. (Refer to dirichlet.mle().) Thus, rather than using a computationally intensive system of equations for a square
      #matrix with K rows and columns to run solve(H,g), as is done in dirichlet.mle() to account for non-symmetric matrices, we can
      #simply use the following for symmetric matrices:
      g <- N * digamma( sum(alpha ) ) - N * digamma(alpha) + N * log.pbar
      b <- N * sirt_digamma1( sum(alpha ))
      a <- (-N*(digamma(alpha+(1e-3))-digamma(alpha-(1e-3)))/(2*(1e-3)))[1]+b
      solution <- numeric(0)
      for(i in 1:length(g)) {
         solution[i] <- -g[i]*(a+(length(g)-2)*b)/((a+(length(g)-1)*b)*(b-a)) - g[i]*(b)/((a+(length(g)-1)*b)*(b-a)) + sum(g*b/((a+(length(g)-1)*b)*(b-a)))
      }
      alpha <- alpha0 - solution
      alpha[ alpha < (1e-10) ] <- 1e-10
      alpha <- rep(mean(alpha),length(alpha)) #Set all concentration parameters to be equal rather than allowing them to freely vary
      conv <- abs( alpha0[1] - alpha[1] )
      #alpha[ alpha < 0 ] <- 1e-10 #Original version of this line, as the function appeared in the sirt package
      alpha <- alpha0 + oldfac*( alpha - alpha0 )
      alpha0 <- alpha
      iter <- iter+1
   }
   alpha0 <- sum(alpha)
   xsi <- alpha / alpha0
   res <- list( alpha=alpha, alpha0=alpha0, xsi=xsi )
   return(res)
}

#' Generate Dirichlet Observations
#'
#' @description This helper function generates a set of observations from a
#'   Dirichlet distribution with underlying concentration parameters equal to
#'   \code{alpha}. This function is heavily based on
#'   \code{\link[gtools]{rdirichlet}}, with modifications to prevent concentration
#'   parameters equal to 0.
#'
#' @param n A numeric value indicating the number of observations to be
#'   generated
#' @param alpha A numeric vector indicating the concentration parameters of the
#'   Dirichlet distribution from which observations should be generated
#' @param zero2 A numeric value representing the minimum allocation permitted
#'   for any given Dirichlet category in order to prevent errors
#' @return A numeric vector with \code{n} Dirichlet observations
#' @export
#' rdirichlet

rdirichlet <- function(n, alpha, zero2=(10^(-255))) { #Based on gtools::rdirichlet
   l<-length(alpha)
   x<-matrix(rgamma(l*n,alpha),ncol=l,byrow=TRUE)
   sm<-as.vector(x%*%rep(1,l)) #Modified from the original function in sirt
   sm[ sm < (zero2) ] <- zero2 #This line was not in the original function in sirt; changes sm values from 0 to "zero2" as needed
   values <- x/as.vector(sm)
   return(values)
}

#' Summary (brittnu)
#'
#' @description This method prints a summary of the Britt's nu reliability
#'   values using an object of class \code{brittnu_rel} outputted from
#'   \link{brittnu}.
#'
#' @param object An object of class \code{brittnu_rel} outputted from
#'   \link{brittnu}
#' @param element The specific reliability coefficient to be printed; permitted
#'   values are \code{"singleomnibus"}, \code{"singlepairwise"},
#'   \code{"multipleomnibus"}, \code{"multiplepairwise"}, \code{"all"}, and
#'   \code{NA}
#' @param ... Other arguments inherited from the generic \code{summary} function
#' @method summary brittnu_rel
#' @exportS3Method summary brittnu_rel
#' @export
#' summary.brittnu_rel

summary.brittnu_rel <- function(object, element=NA, ...) {
   return_value <- "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
   if(element %in% c("singleomnibus", "singlepairwise", "multipleomnibus", "multiplepairwise", "all", NA)) {
      if(is.na(object$type)) {
         row = "Observation "
      } else if(object$type == "wt") {
         row = "Topic "
      } else if(object$type == "td") {
         row = "Document "
      }
      if(is.na(element)) {
         return_value <- paste0(return_value, "\nMultiple-Observation Omnibus Britt's Nu\n", format(object$multipleomnibus, nsmall=10))
         if(length(object$warnings[[3]]) > 0) {
            return_value <- paste0(return_value, "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nWarnings:")
            for(i in 1:length(object$warnings[[3]])) {
               return_value <- paste0(return_value, "\n", i, ". ", object$warnings[[3]][i])
            }
         }
         return_value <- paste0(return_value, "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nTo retrieve all reliabilities, call summary() with element='all'.")
         return_value <- paste0(return_value, "\nTo retrieve a specific type of reliability, you can set\nelement='singleomnibus', element='singlepairwise',\nelement='multipleomnibus', or element='multiplepairwise'.\n")
      } else if(element=="singleomnibus") {
         return_value <- paste0(return_value, "\nSingle-Observation Omnibus Britt's Nu")
         for(i in 1:length(object$singleomnibus)) {
            return_value <- paste0(return_value, "\n", row, i, ": ", format(object$singleomnibus[i], nsmall=10, width=15-nchar(i)+nchar(length(object$singleomnibus))))
         }
         if(length(object$warnings[[1]]) > 0) {
            return_value <- paste0(return_value, "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nWarnings:")
            for(i in 1:length(object$warnings[[1]])) {
               return_value <- paste0(return_value, "\n", i, ". ", object$warnings[[1]][i])
            }
         }
         return_value <- paste0(return_value, "\n")
      } else if(element=="singlepairwise") {
         return_value <- paste0(return_value, "\nSingle-Observation Pairwise Britt's Nu")
         for(i in 1:(length(object$singlepairwise))) {
            for(j in (i+1):(length(object$singlepairwise[[i]]))) {
               if(j > 2) {
                  return_value <- paste0(return_value, "\n")
               }
               return_value <- paste0(return_value, "\nRaters ", i, " and ", j)
               for(k in 1:length(object$singlepairwise[[i]][[j]])) {
                  return_value <- paste0(return_value, "\n", row, k, ": ", format(object$singlepairwise[[i]][[j]][k], nsmall=10, width=13-nchar(k)+nchar(length(object$singlepairwise[[i]][[j]]))))
               }
            }
         }
         return_value <- paste0(return_value, "\n")
         if(length(object$assignments) > 1) {
            return_value <- paste0(return_value, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nMatrix of Reordered Topics:\n")
            cat(return_value)
            print(object$assignments)
            return_value <- character(0)
         }
         if(length(object$warnings[[2]]) > 0) {
            return_value <- paste0(return_value, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nWarnings:")
            for(i in 1:length(object$warnings[[2]])) {
               return_value <- paste0(return_value, "\n", i, ". ", object$warnings[[2]][i])
            }
            return_value <- paste0(return_value, "\n")
         }
      } else if(element=="multipleomnibus") {
         return_value <- paste0(return_value, "\nMultiple-Observation Omnibus Britt's Nu")
         return_value <- paste0(return_value, "\n", format(object$multipleomnibus, nsmall=10))
         if(length(object$warnings[[3]]) > 0) {
            return_value <- paste0(return_value, "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nWarnings:")
            for(i in 1:length(object$warnings[[3]])) {
               return_value <- paste0(return_value, "\n", i, ". ", object$warnings[[3]][i])
            }
         }
         return_value <- paste0(return_value, "\n")
      } else if(element=="multiplepairwise") {
         return_value <- paste0(return_value, "\nMultiple-Observation Pairwise Britt's Nu")
         for(i in 1:(length(object$multiplepairwise))) {
            for(j in (i+1):(length(object$multiplepairwise[[i]]))) {
               return_value <- paste0(return_value, "\nRaters ", i, " and ", j, ": ", format(object$multiplepairwise[[i]][[j]], nsmall=10, width=13+nchar(length(object$multiplepairwise)-1)+nchar(length(object$multiplepairwise))-nchar(i)-nchar(j)))
            }
         }
         return_value <- paste0(return_value, "\n")
         if(length(object$assignments) > 1) {
            return_value <- paste0(return_value, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nMatrix of Reordered Topics:\n")
            cat(return_value)
            print(object$assignments)
            return_value <- character(0)
         }
         if(length(object$warnings[[4]]) > 0) {
            return_value <- paste0(return_value, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nWarnings:")
            for(i in 1:length(object$warnings[[4]])) {
               return_value <- paste0(return_value, "\n", i, ". ", object$warnings[[4]][i])
            }
            return_value <- paste0(return_value, "\n")
         }
      } else if(element=="all") {
         return_value <- paste0(return_value, "\nMultiple-Observation Omnibus Britt's Nu")
         return_value <- paste0(return_value, "\n", format(object$multipleomnibus, nsmall=10))
         return_value <- paste0(return_value, "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
         return_value <- paste0(return_value, "\nSingle-Observation Omnibus Britt's Nu")
         for(i in 1:length(object$singleomnibus)) {
            return_value <- paste0(return_value, "\n", row, i, ": ", format(object$singleomnibus[i], nsmall=10, width=13-nchar(i)+nchar(length(object$singleomnibus))))
         }
         return_value <- paste0(return_value, "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
         return_value <- paste0(return_value, "\nMultiple-Observation Pairwise Britt's Nu")
         for(i in 1:(length(object$multiplepairwise))) {
            for(j in (i+1):(length(object$multiplepairwise[[i]]))) {
               return_value <- paste0(return_value, "\nRaters ", i, " and ", j, ": ", format(object$multiplepairwise[[i]][[j]], nsmall=10, width=13+nchar(length(object$multiplepairwise)-1)+nchar(length(object$multiplepairwise))-nchar(i)-nchar(j)))
            }
         }
         return_value <- paste0(return_value, "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
         return_value <- paste0(return_value, "\nSingle-Observation Pairwise Britt's Nu")
         for(i in 1:(length(object$singlepairwise))) {
            for(j in (i+1):(length(object$singlepairwise[[i]]))) {
               if(j > 2) {
                  return_value <- paste0(return_value, "\n")
               }
               return_value <- paste0(return_value, "\nRaters ", i, " and ", j)
               for(k in 1:length(object$singlepairwise[[i]][[j]])) {
                  return_value <- paste0(return_value, "\n", row, k, ": ", format(object$singlepairwise[[i]][[j]][k], nsmall=10, width=13-nchar(k)+nchar(length(object$singlepairwise[[i]][[j]]))))
               }
            }
         }
         return_value <- paste0(return_value, "\n")
         if(length(object$assignments) > 1) {
            return_value <- paste0(return_value, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nMatrix of Reordered Topics:\n")
            cat(return_value)
            print(object$assignments)
            return_value <- character(0)
         }
         if(length(object$warnings[[5]]) > 0) {
            return_value <- paste0(return_value, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nWarnings:")
            for(i in 1:length(object$warnings[[5]])) {
               return_value <- paste0(return_value, "\n", i, ". ", object$warnings[[5]][i])
            }
            return_value <- paste0(return_value, "\n")
         }
      }
   } else {
      stop("'element' must be set to 'singleomnibus', 'singlepairwise', 'multipleomnibus',\n'multiplepairwise', 'all', or NA.")
   }
   return_value <- paste0(return_value, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
   cat(return_value)
}

#' Summary (euclidkrip)
#'
#' @description This method prints a summary of the Euclidean Krippendorff's
#'   alpha reliability values using an object of class \code{euclidkrip_rel}
#'   outputted from \link{euclidkrip}.
#'
#' @param object An object of class \code{euclidkrip_rel} outputted from
#'   \link{euclidkrip}
#' @param element The specific reliability coefficient to be printed; permitted
#'   values are \code{"singleomnibus"}, \code{"singlepairwise"},
#'   \code{"multipleomnibus"}, \code{"multiplepairwise"}, \code{"all"}, and
#'   \code{NA}
#' @param ... Other arguments inherited from the generic \code{summary} function
#' @method summary euclidkrip_rel
#' @exportS3Method summary euclidkrip_rel
#' @export
#' summary.euclidkrip_rel

summary.euclidkrip_rel <- function(object, element=NA, ...) {
   return_value <- "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
   if(element %in% c("singleomnibus", "singlepairwise", "multipleomnibus", "multiplepairwise", "all", NA)) {
      row = "Observation "
      if(is.na(element)) {
         return_value <- paste0(return_value, "\nMultiple-Observation Omnibus Euclidean Krippendorff's Alpha\n", format(object$multipleomnibus, nsmall=10))
         if(length(object$warnings[[3]]) > 0) {
            return_value <- paste0(return_value, "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nWarnings:")
            for(i in 1:length(object$warnings[[3]])) {
               return_value <- paste0(return_value, "\n", i, ". ", object$warnings[[3]][i])
            }
         }
         return_value <- paste0(return_value, "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nTo retrieve all reliabilities, call summary() with element='all'.")
         return_value <- paste0(return_value, "\nTo retrieve a specific type of reliability, you can set\nelement='singleomnibus', element='singlepairwise',\nelement='multipleomnibus', or element='multiplepairwise'.\n")
      } else if(element=="singleomnibus") {
         return_value <- paste0(return_value, "\nSingle-Observation Omnibus Euclidean Krippendorff's Alpha")
         for(i in 1:length(object$singleomnibus)) {
            return_value <- paste0(return_value, "\n", row, i, ": ", format(object$singleomnibus[i], nsmall=10, width=15-nchar(i)+nchar(length(object$singleomnibus))))
         }
         if(length(object$warnings[[1]]) > 0) {
            return_value <- paste0(return_value, "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nWarnings:")
            for(i in 1:length(object$warnings[[1]])) {
               return_value <- paste0(return_value, "\n", i, ". ", object$warnings[[1]][i])
            }
         }
         return_value <- paste0(return_value, "\n")
      } else if(element=="singlepairwise") {
         return_value <- paste0(return_value, "\nSingle-Observation Pairwise Euclidean Krippendorff's Alpha")
         for(i in 1:(length(object$singlepairwise))) {
            for(j in (i+1):(length(object$singlepairwise[[i]]))) {
               if(j > 2) {
                  return_value <- paste0(return_value, "\n")
               }
               return_value <- paste0(return_value, "\nRaters ", i, " and ", j)
               for(k in 1:length(object$singlepairwise[[i]][[j]])) {
                  return_value <- paste0(return_value, "\n", row, k, ": ", format(object$singlepairwise[[i]][[j]][k], nsmall=10, width=13-nchar(k)+nchar(length(object$singlepairwise[[i]][[j]]))))
               }
            }
         }
         return_value <- paste0(return_value, "\n")
         if(length(object$assignments) > 1) {
            return_value <- paste0(return_value, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nMatrix of Reordered Topics:\n")
            cat(return_value)
            print(object$assignments)
            return_value <- character(0)
         }
         if(length(object$warnings[[2]]) > 0) {
            return_value <- paste0(return_value, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nWarnings:")
            for(i in 1:length(object$warnings[[2]])) {
               return_value <- paste0(return_value, "\n", i, ". ", object$warnings[[2]][i])
            }
            return_value <- paste0(return_value, "\n")
         }
      } else if(element=="multipleomnibus") {
         return_value <- paste0(return_value, "\nMultiple-Observation Omnibus Euclidean Krippendorff's Alpha")
         return_value <- paste0(return_value, "\n", format(object$multipleomnibus, nsmall=10))
         if(length(object$warnings[[3]]) > 0) {
            return_value <- paste0(return_value, "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nWarnings:")
            for(i in 1:length(object$warnings[[3]])) {
               return_value <- paste0(return_value, "\n", i, ". ", object$warnings[[3]][i])
            }
         }
         return_value <- paste0(return_value, "\n")
      } else if(element=="multiplepairwise") {
         return_value <- paste0(return_value, "\nMultiple-Observation Pairwise Euclidean Krippendorff's Alpha")
         for(i in 1:(length(object$multiplepairwise))) {
            for(j in (i+1):(length(object$multiplepairwise[[i]]))) {
               return_value <- paste0(return_value, "\nRaters ", i, " and ", j, ": ", format(object$multiplepairwise[[i]][[j]], nsmall=10, width=13+nchar(length(object$multiplepairwise)-1)+nchar(length(object$multiplepairwise))-nchar(i)-nchar(j)))
            }
         }
         return_value <- paste0(return_value, "\n")
         if(length(object$assignments) > 1) {
            return_value <- paste0(return_value, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nMatrix of Reordered Topics:\n")
            cat(return_value)
            print(object$assignments)
            return_value <- character(0)
         }
         if(length(object$warnings[[4]]) > 0) {
            return_value <- paste0(return_value, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nWarnings:")
            for(i in 1:length(object$warnings[[4]])) {
               return_value <- paste0(return_value, "\n", i, ". ", object$warnings[[4]][i])
            }
            return_value <- paste0(return_value, "\n")
         }
      } else if(element=="all") {
         return_value <- paste0(return_value, "\nMultiple-Observation Omnibus Euclidean Krippendorff's Alpha")
         return_value <- paste0(return_value, "\n", format(object$multipleomnibus, nsmall=10))
         return_value <- paste0(return_value, "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
         return_value <- paste0(return_value, "\nSingle-Observation Omnibus Euclidean Krippendorff's Alpha")
         for(i in 1:length(object$singleomnibus)) {
            return_value <- paste0(return_value, "\n", row, i, ": ", format(object$singleomnibus[i], nsmall=10, width=13-nchar(i)+nchar(length(object$singleomnibus))))
         }
         return_value <- paste0(return_value, "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
         return_value <- paste0(return_value, "\nMultiple-Observation Pairwise Euclidean Krippendorff's Alpha")
         for(i in 1:(length(object$multiplepairwise))) {
            for(j in (i+1):(length(object$multiplepairwise[[i]]))) {
               return_value <- paste0(return_value, "\nRaters ", i, " and ", j, ": ", format(object$multiplepairwise[[i]][[j]], nsmall=10, width=13+nchar(length(object$multiplepairwise)-1)+nchar(length(object$multiplepairwise))-nchar(i)-nchar(j)))
            }
         }
         return_value <- paste0(return_value, "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
         return_value <- paste0(return_value, "\nSingle-Observation Pairwise Euclidean Krippendorff's Alpha")
         for(i in 1:(length(object$singlepairwise))) {
            for(j in (i+1):(length(object$singlepairwise[[i]]))) {
               if(j > 2) {
                  return_value <- paste0(return_value, "\n")
               }
               return_value <- paste0(return_value, "\nRaters ", i, " and ", j)
               for(k in 1:length(object$singlepairwise[[i]][[j]])) {
                  return_value <- paste0(return_value, "\n", row, k, ": ", format(object$singlepairwise[[i]][[j]][k], nsmall=10, width=13-nchar(k)+nchar(length(object$singlepairwise[[i]][[j]]))))
               }
            }
         }
         return_value <- paste0(return_value, "\n")
         if(length(object$assignments) > 1) {
            return_value <- paste0(return_value, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nMatrix of Reordered Topics:\n")
            cat(return_value)
            print(object$assignments)
            return_value <- character(0)
         }
         if(length(object$warnings[[5]]) > 0) {
            return_value <- paste0(return_value, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nWarnings:")
            for(i in 1:length(object$warnings[[5]])) {
               return_value <- paste0(return_value, "\n", i, ". ", object$warnings[[5]][i])
            }
            return_value <- paste0(return_value, "\n")
         }
      }
   } else {
      stop("'element' must be set to 'singleomnibus', 'singlepairwise', 'multipleomnibus',\n'multiplepairwise', 'all', or NA.")
   }
   return_value <- paste0(return_value, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
   cat(return_value)
}

#' Print (brittnu)
#'
#' @description This helper method calls \code{summary()} when the user attempts
#'   to print an object of class \code{brittnu_rel} outputted from
#'   \link{brittnu}.
#'
#' @param x An object of class \code{brittnu_rel} outputted from
#'   \link{brittnu}
#' @param element The specific reliability coefficient to be printed; permitted
#'   values are \code{"singleomnibus"}, \code{"singlepairwise"},
#'   \code{"multipleomnibus"}, \code{"multiplepairwise"}, \code{"all"}, and
#'   \code{NA}
#' @param ... Other arguments inherited from the generic \code{print} function
#' @method print brittnu_rel
#' @exportS3Method print brittnu_rel
#' @export
#' print.brittnu_rel

print.brittnu_rel <- function(x, element=NA, ...) {
   summary(x, element=element)
}

#' Print (euclidkrip)
#'
#' @description This helper method calls \code{summary()} when the user attempts
#'   to print an object of class \code{euclidkrip_rel} outputted from
#'   \link{euclidkrip}.
#'
#' @param x An object of class \code{euclidkrip_rel} outputted from
#'   \link{euclidkrip}
#' @param element The specific reliability coefficient to be printed; permitted
#'   values are \code{"singleomnibus"}, \code{"singlepairwise"},
#'   \code{"multipleomnibus"}, \code{"multiplepairwise"}, \code{"all"}, and
#'   \code{NA}
#' @param ... Other arguments inherited from the generic \code{print} function
#' @method print euclidkrip_rel
#' @exportS3Method print euclidkrip_rel
#' @export
#' print.euclidkrip_rel

print.euclidkrip_rel <- function(x, element=NA, ...) {
   summary(x, element=element)
}