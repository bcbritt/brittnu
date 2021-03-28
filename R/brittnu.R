#' Britt's Nu for Dirichlet-Distributed Data
#'
#' @description This function computes the Britt's nu reliability coefficient
#' for data sets composed of multiple Dirichlet-distributed deviates, as
#' described by Britt (under review).
#'
#' @details When estimating Britt's nu, it is common to compute four separate
#' coefficients in this family: Britt's single-distribution nu for all
#' cross-validation iterations, Britt's single-distribution nu for pairs of
#' iterations, Britt's omnibus nu for all cross-validation iterations, and
#' Britt's single-distribution nu for pairs of iterations. The versions that use
#' all cross-validation iterations are more vital to the assessment of
#' reliability, but the pairwise versions are sometimes useful as diagnostic
#' tools.
#'
#' By default, \code{brittnu()} provides all four versions of Britt's nu. Since
#' computing the pairwise versions of Britt's nu requires more time than using
#' all cross-validation iterations, the pairwise versions may optionally be
#' omitted from the computation by specifying \code{pairwise = FALSE}.
#'
#' The expected difference component of Britt's nu is estimated using a large
#' number of samples from Dirichlet distributions with the same concentration
#' parameters as the original data set. In some cases, this process can become
#' computationally intensive. For those instances, it is advisable to set
#' \code{lower_bound = TRUE}, which uses i.i.d. beta distributions to estimate
#' the lower bound of the expected differences between observations, ultimately
#' yielding an estimated lower bound for Britt's nu itself.
#'
#' One of the most important uses for Britt's nu is to assess the reliability of
#' the allocations of words to topics or of topics to documents via latent
#' Dirichlet allocation (LDA). When the results of multiple LDA cross-validation
#' iterations are provided as \code{x}, the \code{type} parameter must be set to
#' either \code{"wt"} or \code{"td"} to indicate whether word-topic or
#' topic-document reliability, respectively, should be assessed.
#'
#' Crucially, different LDA cross-validation iterations may result in the same
#' topics appearing in a different sequence. As such, those topics may need to
#' be reordered in order to yield an optimal fit. In practice, it is generally
#' infeasible to assess every possible combination of topic across all
#' cross-validation iterations. As such, \code{brittnu()} provides two methods
#' of converging toward an optimal topic order for each iteration. The default,
#' \code{method = "rotational"}, takes the provided sequence and swaps
#' pairs of categories until no further swaps would further improve the fit.
#' Alternatively, \code{method = "forward"} (in development) assigns the
#' best-fitting category, one at a time, until all categories have been assigned
#' to all distributions. The forward method is more computationally efficient
#' for use cases with many categories and cross-validation iterations, but it is
#' likely to generate a worse fit than the rotational method. On the other hand,
#' the greater reliability is, the better the forward method is expected to
#' perform, and when reliability is poor, it may not be important to precisely
#' estimate Britt's nu.
#' 
#' Additional usage notes:
#'
#' 1. When importing your own data set (rather than a list of objects outputted
#' from \code{\link[topicmodels]{LDA}}), each row of the data set should be a
#' distribution (e.g., an LDA topic), and each column should be a category
#' within that distribution (e.g., a word).
#'
#' 2. It is generally recommended that \code{estimate_pairwise_alpha_from_joint}
#' be set to \code{TRUE}, which is the default setting. If concentration
#' parameters must be estimated from the data, and if all cross-validation
#' iterations used the same data set or are otherwise assumed to have emerged
#' from the same underlying distribution, then there is little reason to expect
#' different sets of concentration parameters to be necessary across iterations.
#' You should only consider setting \code{estimate_pairwise_alpha_from_joint} to
#' \code{FALSE} if different data sets that may not have come from the same
#' underlying distribution were used for different cross-validation iterations.
#'
#' 3. If you are assessing the reliability of the allocations of words to topics
#  via LDA, each cross-validation iteration may optionally use a different set
#' set of documents. For instance, if \code{mydata1}, \code{mydata2},
#' \code{mydata3}, \code{mydata4}, and \code{mydata5} each contain 20% of the
#' documents from a single data set, then you could run
#'
#' \preformatted{cv1 <- LDA(mydata1, k=15)
#' cv2 <- LDA(mydata2, k=15)
#' cv3 <- LDA(mydata3, k=15)
#' cv4 <- LDA(mydata4, k=15)
#' cv5 <- LDA(mydata5, k=15)
#' cv <- list(cv1, cv2, cv3, cv4, cv5)
#' rel <- brittnu(cv, type="wt", shuffle=TRUE, different_documents=TRUE)
#' print(rel)}
#'
#' to assess the word-topic allocation reliability. Any words that appear in
#' some cross-validation iterations but not others will automatically be set
#' to an allocation near 0 for any iterations in which they are absent from the
#' data set. (In other words, if each topic in that iteration represents its own
#' Dirichlet distribution, the missing word will be treated as a category with a
#' concentration parameter and allocation of zero.) If this applies in your
#' case, set \code{different_documents = TRUE} when conducting your LDA. Note,
#' however, that if you are assessing the reliability of the allocations of
#' topics to documents, then each document represents its own respective
#' Dirichlet distribution, and any LDA in which a given document does not appear
#' will simply be excluded from the reliability computation for that document,
#' such that there is little benefit in using different sets of documents for
#' different cross-validation iterations. Therefore, when assessing
#' topic-document reliability, it is advised that all documents be included in
#' all cross-validation iterations.
#'
#' 4. For Dirichlet distributions with many categories and function calls with a
#' large value for the samples parameter, setting \code{lower_bound=TRUE} is
#' strongly recommended, as shuffling numerous samples with a large number of
#' categories may require an unreasonable amount of time. Additionally, for
#' Dirichlet distributions with many categories, precise estimates of the
#' concentration parameters may require excessive time, so when possible, a
#' priori known concentration parameters should be provided via the \code{alpha}
#' parameter rather than estimating them from the data. When this is not
#' possible, it may be advisable to change \code{convcrit} from its default
#' value (0.00001) to a larger number. (With approximately 10,000 categories, a
#' single iteration of the convergence algorithm to estimate concentration
#' parameters may take several minutes to complete.) You may also wish to set
#' \code{verbose = TRUE} to receive periodic progress updates and ensure that
#' the computation is proceeding as expected.
#'
#' @param x A list with each entry being a single cross-validation iteration;
#'   each list entry should be either the output from a
#'   \code{\link[topicmodels]{LDA}} function call or a 2D double vector with
#'   distributions as the rows and each category within each distribution as the
#'   columns (such that each row in the vector sums to 1)
#' @param type A string indicating the type of reliability being assessed
#'   (either word-topic or topic-document); if \code{x} contains the output from
#'   multiple \code{\link[topicmodels]{LDA}} function calls, this must be set to
#'   either \code{"wt"} (word-topic reliability) or \code{"td"} (topic-document
#'   reliability)
#' @param alpha A list whose length is equal to the number of distributions,
#'   with each list element containing a vector of length K comprising the
#'   concentration parameters for the K categories in that distribution; if this
#'   is \code{NA}, the concentration parameters will be estimated from \code{x}
#' @param estimate_pairwise_alpha_from_joint A boolean value indicating, when
#'   estimating reliability for pairs of cross-validation iterations, whether
#'   each pair of iterations should use the concentration parameters estimated
#'   across all iterations (\code{TRUE}) or whether separate sets of
#'   concentration parameters should be estimated for each pair of
#'   cross-validation iterations (\code{FALSE}); this parameter only has an
#'   effect when \code{alpha=NA}, and it is strongly suggested that the default
#'   value of \code{TRUE} be used for this parameter
#' @param symmetric_alpha A boolean value indicating whether all concentration
#'   parameters in each cross-validation iteration should be assumed to be
#'   equal, which is common in LDA; this parameter only has an effect when
#'   \code{alpha=NA}
#' @param shuffle A boolean value indicating whether or not the topics in each
#'   iteration may have been shuffled in the context of LDA; if this is
#'   \code{TRUE} and \code{type="wt"} or \code{type=NA}, then the distributions
#'   within each iteration will be reordered to achieve a (local) optimal fit,
#'   whereas if this is \code{TRUE} and \code{type="td",} the categories within
#'   each distribution will be reordered instead, so if you want the categories
#'   within each distribution to be reordered, then you should set
#'   \code{type="td"} even if \code{x} contains raw data rather than the output
#'   from multiple \code{\link[topicmodels]{LDA}} function calls
#' @param method A string indicating what method (\code{"rotational"} or
#'   \code{"forward"}) will be used to reorder topics (see Britt, under review);
#'   the \code{"forward"} method will be implemented at a later date
#' @param different_documents A boolean value indicating, if each list element
#'   in \code{x} represents the output from an \code{\link[topicmodels]{LDA}}
#'   function call, whether some cross-validation iterations used different
#'   sets of documents than others
#' @param pairwise A boolean value indicating whether pairwise reliability
#'   between individual cross-validation iterations should be computed; if
#'   \code{FALSE}, pairwise reliability will be ignored in order to reduce the
#'   time and memory complexity of the computation
#' @param samples An integer value indicating how many samples to use to
#'   estimate expected differences when \code{shuffle=TRUE}
#' @param lower_bound A boolean value indicating whether the lower bound of the
#'   expected differences (based on i.i.d. beta distributions) should be used
#'   rather than shuffling Dirichlet distributions to estimate the exact
#'   difference in every sample when \code{shuffle=TRUE}; for Dirichlet
#'   distributions with many categories and function calls with a large value of
#'   samples, setting this to \code{TRUE} is strongly recommended, as reordering
#'   numerous samples with a large number of categories may require an
#'   unreasonable amount of time
#' @param convcrit A numeric value indicating the threshold for convergence used
#'   to estimate concentration parameters when \code{alpha=NA}
#' @param maxit A numeric value indicating the maximum number of iterations used
#'   to estimate concentration parameters when \code{alpha=NA}
#' @param verbose A boolean value indicating whether \code{brittnu()} and its
#'   helper functions should provide progress updates
#' @param zero A numeric value; if \code{type="wt"} and
#'   \code{different_documents=TRUE}, then whenever a word appears in some
#'   cross-validation iterations but not others, in order to prevent errors,
#'   this value is assigned as its allocation to all topics within any cross-
#'   validation iteration in which the word did not originally appear
#' @param zero2 A numeric value; if \code{alpha=NA}, then in order to prevent
#'   errors when concentration parameters are being estimated, this is the
#'   minimum allocation permitted for any given category during the estimation
#'   procedure
#' @return A list with four elements: Britt's single-distribution nu for all
#'   cross-validation iterations (as a numeric vector of length K indicating the
#'   reliability for each of the K categories), Britt's single-distribution nu
#'   for pairs of cross-validation iterations (as a list containing lists
#'   containing numeric vectors of length K, e.g., the second element of the
#'   first list contains the reliability for cross-validation iterations 1 and
#'   2), Britt's omnibus nu for all cross-validation iterations (as a numeric
#'   value), and Britt's omnibus nu for pairs of cross-validation iterations (as
#'   a list containing lists containing numeric values, e.g., the second element
#'   of the first list contains the reliability for cross-validation iterations
#'   1 and 2)
#' @importFrom "utils" "flush.console"
#' @importFrom "stats" "rgamma"
#' @importFrom "methods" "slot"
#' @importFrom "plyr" "rbind.fill.matrix"
#' @section References:
#'   Britt, B. C. (under review). An interrater reliability coefficient for
#'   beta-distributed and Dirichlet-distributed data.
#' @examples
#' require(topicmodels)
#' data("AssociatedPress")
#' set.seed(1797)
#' ap1 <-  LDA(AssociatedPress, k = 40)
#' set.seed(1798)
#' ap2 <-  LDA(AssociatedPress, k = 40)
#' set.seed(1799)
#' ap3 <-  LDA(AssociatedPress, k = 40)
#' set.seed(1800)
#' ap4 <-  LDA(AssociatedPress, k = 40)
#' set.seed(1801)
#' ap5 <-  LDA(AssociatedPress, k = 40)
#' set.seed(1802)
#' ap6 <-  LDA(AssociatedPress, k = 40)
#' set.seed(1803)
#' ap7 <-  LDA(AssociatedPress, k = 40)
#' set.seed(1804)
#' ap8 <-  LDA(AssociatedPress, k = 40)
#' set.seed(1805)
#' ap9 <-  LDA(AssociatedPress, k = 40)
#' set.seed(1806)
#' ap10 <-  LDA(AssociatedPress, k = 40)
#' set.seed(1807)
#' ap11 <-  LDA(AssociatedPress, k = 40)
#' set.seed(1808)
#' ap12 <-  LDA(AssociatedPress, k = 40)
#' set.seed(1809)
#' ap13 <-  LDA(AssociatedPress, k = 40)
#' set.seed(1810)
#' ap14 <-  LDA(AssociatedPress, k = 40)
#' set.seed(1811)
#' ap15 <-  LDA(AssociatedPress, k = 40)
#' set.seed(1812)
#' ap16 <-  LDA(AssociatedPress, k = 40)
#' set.seed(1813)
#' ap17 <-  LDA(AssociatedPress, k = 40)
#' set.seed(1814)
#' ap18 <-  LDA(AssociatedPress, k = 40)
#' set.seed(1815)
#' ap19 <-  LDA(AssociatedPress, k = 40)
#' set.seed(1816)
#' ap20 <-  LDA(AssociatedPress, k = 40)
#' ap_all_40 <- list(ap1,ap2,ap3,ap4,ap5,ap6,ap7,ap8,ap9,ap10,ap11,ap12,ap13,
#'                   ap14,ap15,ap16,ap17,ap18,ap19,ap20)
#' set.seed(1817)
#' reliability_40_td <- brittnu(ap_all_40, type="td", symmetric_alpha=TRUE,
#'                              shuffle=TRUE, samples=1000)
#' #Britt's single-distribution nu for all iterations
#' print(reliability_40_td[[1]])
#' #Britt's single-distribution nu for pairs of iterations
#' print(reliability_40_td[[2]])
#' #Britt's omnibus nu for all iterations
#' print(reliability_40_td[[3]])
#' #Britt's omnibus nu for pairs of iterations
#' print(reliability_40_td[[4]])
#' #All Britt's nu values for topic-document reliability
#' print(reliability_40_td)
#' set.seed(1820)
#' reliability_40_wt <- brittnu(ap_all_40, type="wt", symmetric_alpha=TRUE,
#'                              shuffle=TRUE, samples=1000)
#' #Britt's single-distribution nu for all iterations
#' print(reliability_40_wt[[1]])
#' #Britt's single-distribution nu for pairs of iterations
#' print(reliability_40_wt[[2]])
#' #Britt's omnibus nu for all iterations
#' print(reliability_40_wt[[3]])
#' #Britt's omnibus nu for pairs of iterations
#' print(reliability_40_wt[[4]])
#' #All Britt's nu values for word-topic reliability
#' print(reliability_40_wt)
#' @export
#' brittnu

brittnu <- function(x, type=NA, alpha=NA,
                    estimate_pairwise_alpha_from_joint=TRUE,
                    symmetric_alpha=FALSE, shuffle=FALSE, method="rotational",
                    different_documents=FALSE, pairwise=TRUE, samples=1000,
                    lower_bound=FALSE, convcrit=0.00001,  maxit=1000,
                    verbose=FALSE, zero=(10^(-16)), zero2=(10^(-255))) {
   raw_alpha <- NA
   if(!is.na(type)) {
      if(!(type %in% c("wt","td"))) {
         warning("You must set type='wt', type='td', or type=NA. Since an invalid value was set,
type will be treated as NA.", call. = FALSE)
         type <- NA
      }
   }
   if((class(x[[1]]) == "LDA_VEM") | (class(x[[1]]) == "LDA_Gibbs")) {
      if(!(type %in% c("wt","td"))) {
         stop("If x is a list of S4 objects outputted from topicmodels::LDA(), then you
must specify the type argument as either 'wt' (to assess the reliability of word
allocations to different topics) or 'td' (to assess the reliability of topic
allocations to different documents).", call. = FALSE)
      }
      if(!shuffle) {
         warning("If x is a list of S4 objects outputted from topicmodels::LDA(), then each of
those S4 objects may have its distributions in a different order due to the
random seeds used to conduct the analyses. It is strongly recommended that you
set the shuffle argument to TRUE when assessing reliability for this data set.", call. = FALSE)
      }
      if(type=="td") {
         if(identical(all.equal(alpha,NA),TRUE)) {
            #If the alpha values from topicmodels::LDA() will be useful, retain
            #them for later use
            raw_alpha <- vector("list",length(x))
            for(i in 1:length(raw_alpha)) {
               if(verbose) {
                  message(paste("Retaining alpha values from 'x': Step", i,
                                "of", length(raw_alpha)))
                  utils::flush.console()
               }
               raw_alpha[[i]] <- methods::slot(x[[i]], "alpha")
            }
         }
         for(i in 1:length(x)) {
            #Each list entry becomes an M x K double vector, with each row
            #summing to 1
            x[[i]] <- methods::slot(x[[i]], "gamma")
         }
         if(different_documents) {
            #If different documents were used in different cross-validation
            #iterations, then in any iteration for which a given document is
            #absent, all of its allocations must be set to NA (since setting
            #them to 0 would compromise reliability and would also not
            #constitute a valid Dirichlet distribution)
            names <- character(0)
            for(i in 1:length(x)) {
               names <- union(names, methods::slot(x[[i]], "documents"))
            }
            named_data <- matrix(NA,0,length(names))
            colnames(named_data) <- names
            for(i in 1:length(x)) {
               if(verbose) {
                  message(paste("Replacing missing values in 'x' with 'zero':",
                                "Step", i, "of", length(x)))
                  utils::flush.console()
               }
               temp_data <- exp(methods::slot(x[[i]], "gamma"))
               colnames(temp_data) <- methods::slot(x[[i]], "documents")
               x[[i]] <- plyr::rbind.fill.matrix(named_data, temp_data)
               colnames(x[[i]]) <- NULL
            }
         }
      }
      if(type=="wt") {
         if(different_documents) {
            #If different documents were used in different cross-validation
            #iterations, then in any iteration for which a given word is absent,
            #its allocation must be set to a value extremely close to zero
            #(since setting them to exactly zero would not yield an appropriate
            #Dirichlet distribution)
            names <- character(0)
            for(i in 1:length(x)) {
               names <- union(names, methods::slot(x[[i]], "terms"))
            }
            named_data <- matrix(0,0,length(names))
            colnames(named_data) <- names
            for(i in 1:length(x)) {
               if(verbose) {
                  message(paste("Replacing missing values in 'x' with 'zero':",
                                "Step", i, "of", length(x)))
                  utils::flush.console()
               }
               temp_data <- exp(methods::slot(x[[i]], "beta"))
               colnames(temp_data) <- methods::slot(x[[i]], "terms")
               x[[i]] <- plyr::rbind.fill.matrix(named_data, temp_data)
               x[[i]][is.na(x[[i]])] <- zero
               colnames(x[[i]]) <- NULL
            }
         } else {
            for(i in 1:length(x)) {
               x[[i]] <- exp(methods::slot(x[[i]], "beta"))
            }
         }
      }
      if(!symmetric_alpha) {
         warning("If x is a list of S4 objects outputted from topicmodels::LDA(), then you are
attempting to assess the reliability of an LDA procedure. In most LDAs, the
prior distribution of words on topics or topics on documents is symmetrical. If
that is the case, then you should set symmetric_alpha=TRUE. Doing so will make
the reliability coefficients more valid, and the algorithm will also run faster.
If you did not use a symmetrical prior distribution, then you should keep
symmetric_alpha=FALSE. However, if you are working with a non-symmetrical prior
distribution and you are estimating the concentration parameters rather than
providing them via the alpha parameter in your call to brittnu(), be forewarned
that the computation may be extraordinarily slow, particularly for larger data
sets. If you are working with a non-symmetrical prior, then if possible, it is
strongly recommended to specify the Dirichlet concentration parameters using the
alpha parameter in the brittnu() function call.", call.=FALSE)
      }
   }

   if(shuffle) {
      if(!identical(all.equal(alpha,NA),TRUE)) { #If alpha is not NA...
         if(length(alpha) > 1) {
            for(i in 2:length(alpha)) {
               if(!identical(all.equal(alpha[[i]],alpha[[1]]),TRUE)) { #If any distributions have different sets of a priori alphas, the distributions cannot be shuffled
                  stop("If categories are allowed to be reordered, then alpha must be set to NA
or a list of vectors in which all values with the same vector index are
identical. Distinct sets of alpha levels corresponding to different
distributions will not be retained when those distributions are shuffled.", call.=FALSE)
               }
            }
         }
      }
      if(method=="forward") {
         stop("Forward reordering has not yet been implemented. Please set
method='rotational' instead.", call.=FALSE)
      }
      if(!(method %in% c("forward","rotational"))) {
         stop("You must set method='rotational' or method='forward'.",
               call.=FALSE)
      }
      if(!is.na(type)) {
         if(type=="td") {
            #If the topics are the categories within a distribution, not the
            #distributions themselves, then it is those categories that need to
            #be shuffled rather than the distributions! I can deal with this by
            #transposing each cross-validation iteration, then retransposing
            #them afterward.
            for(i in 1:length(x)) {
               x[[i]] <- t(x[[i]])
            }
         }
      }
      shuffle_results <- shuffle_distributions(x, method, verbose)
      x <- shuffle_results[[1]] #Change x to be the newly shuffled data set
      assignments <- shuffle_results[[2]]
      rm(shuffle_results)
      if(!is.na(type)) {
         if(type=="td") {
            #If the topics are the categories within a distribution, not the
            #distributions themselves, then it is those categories that need to
            #be shuffled rather than the distributions! I can deal with this by
            #transposing each cross-validation iteration, then retransposing
            #them afterward.
            assignments <- t(assignments)
            for(i in 1:length(x)) {
               x[[i]] <- t(x[[i]])
            }
         }
      }
   }

   #Obtain concentration parameters
   if(identical(all.equal(alpha,NA),TRUE)) {
      #If concentration parameters have not already been specified, they must be
      #obtained
      if(identical(all.equal(type,"td"),TRUE)) {
         #Type of reliability: topic-document
         if(!identical(all.equal(raw_alpha,NA),TRUE)) {
            #If raw_alpha is not NA, then topic-document alphas were already
            #extracted from the results of topicmodels::LDA()
            joint_alpha <- vector("list", dim(x[[1]])[1])
            for(i in 1:length(joint_alpha)) {
               if(verbose) {
                  message(paste("Setting 'joint_alpha' from 'raw_alpha': Step",
                                i, "of", length(joint_alpha)))
                  utils::flush.console()
               }
               joint_alpha[[i]] <- rep(mean(unlist(raw_alpha)),dim(x[[1]])[2])
            }
            if(pairwise) {
               pairwise_alpha <- vector("list",(length(x)-1))
               for(i in 1:(length(pairwise_alpha))) {
                  if(verbose) {
                     message(paste("Setting 'pairwise_alpha' from 'raw_alpha':",
                                   "Step", i, "of", length(pairwise_alpha)))
                     utils::flush.console()
                  }
                  pairwise_alpha[[i]] <- vector("list",length(x))
                  for(j in i:length(pairwise_alpha[[i]])) {
                     if(estimate_pairwise_alpha_from_joint) {
                        pairwise_alpha[[i]][[j]] <- rep(mean(unlist(raw_alpha)),
                                                        dim(x[[1]])[2])
                        for(k in 2:(dim(x[[1]])[1])) {
                           pairwise_alpha[[i]][[j]] <-
                              rbind(pairwise_alpha[[i]][[j]],
                                    rep(mean(unlist(raw_alpha)),dim(x[[1]])[2]))
                        }                     
                     } else {
                        pairwise_alpha[[i]][[j]] <- vector("double")
                        pairwise_alpha[[i]][[j]][1,] <- rep(((raw_alpha[[i]] +
                                                             raw_alpha[[j]])/2),
                                                            dim(x[[1]])[2])
                        for(k in 2:(dim(x[[1]])[1])) {
                           pairwise_alpha[[i]][[j]] <-
                              rbind(pairwise_alpha[[i]][[j]],
                                    rep(((raw_alpha[[i]] + raw_alpha[[j]])/2),
                                         dim(x[[1]])[2]))
                        }
                     }
                  }
               }
            }
         } else { #Concentration parameters must be estimated from the data
            alphas <- estimate_alpha_from_data(x, pairwise,
                                             estimate_pairwise_alpha_from_joint,
                                             symmetric_alpha, convcrit=convcrit,
                                             maxit=maxit, verbose=verbose)
            joint_alpha <- alphas[[1]]
            if(pairwise) {
               pairwise_alpha <- alphas[[2]]
            }
         }
      } else {
         #topic-word alphas are not given by topicmodels::LDA() and therefore
         #must be estimated from the data
         alphas <- estimate_alpha_from_data(x, pairwise,
                                            estimate_pairwise_alpha_from_joint,
                                            symmetric_alpha, convcrit=convcrit,
                                            maxit=maxit, verbose=verbose)
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
         for(i in 1:length(pairwise_alpha)) {
            if(verbose) {
               message(paste("Setting 'pairwise_alpha' from 'alpha': Step", i,
                             "of", length(pairwise_alpha)))
               utils::flush.console()
            }
            pairwise_alpha[[i]] <- vector("list",length(x))
            for(j in 1:length(pairwise_alpha[[i]])) {
               pairwise_alpha[[i]][[j]] <- alpha[[1]]
               for(k in 2:length(alpha)) {
                  pairwise_alpha[[i]][[j]] <- rbind(pairwise_alpha[[i]][[j]],
                                                    alpha[[k]])
               }
            }
         }
      }
   }
   if(!pairwise) {
      #If !pairwise, then pairwise_alpha was never set... however, we still need
      #to instantiate the pairwise_alpha variable so that it doesn't trigger
      #errors when estimate_shuffled_single_expectation() or
      #compute_standard_single_expectation() is eventually called
      pairwise_alpha = NA
   }

   #Get observed and expected differences for each individual distribution
   #First, compute each D_o
   #Note: We have to compute do_single_pairwise regardless of whether or not
   #pairwise=TRUE, as this is used in the subsequent computation of
   #do_single_joint
   do_single_pairwise <- vector("list",length(x))
   for(i in 1:length(do_single_pairwise)) { #For each iteration i...
      do_single_pairwise[[i]] <- vector("list",length(x))
      for(j in 1:length(do_single_pairwise[[i]])) {
         #And for each corresponding iteration j...
         do_single_pairwise[[i]][[j]] <- vector("double")
         #The above vector indicates the sum of squared differences between
         #iterations i and j for each respective distribution
      }
   }
   for(i in 1:(length(do_single_pairwise)-1)) {
      #Compute pairwise D_o value for iterations i and j
      for(j in (i+1):length(do_single_pairwise[[i]])) {
         for(k in 1:nrow(x[[i]])) {
            squared_d <- 0
            for(l in 1:ncol(x[[1]])) {
               squared_d <- squared_d + ((x[[i]][k,l] - x[[j]][k,l])^2)
            }
            do_single_pairwise[[i]][[j]][k] <- squared_d
         }
      }
   }
   do_single_joint <- vector("double")
   for(i in 1:nrow(x[[1]])) {
      #For each distribution, find the sum of all pairwise D_o values for that
      #distribution
      do_single_joint[i] <- 0
      for(j in 1:(length(x)-1)) {
         for(k in (j+1):length(do_single_pairwise[[j]])) {
            do_single_joint[i] <- do_single_joint[i] +
                                  do_single_pairwise[[j]][[k]][i]
         }
      }
   }
   do_single_joint <- do_single_joint * 2 / (length(x) * (length(x)-1))

   #Determine each D_e
   if(shuffle) {
      #Use the modified formula to compute the joint D_e for each distribution
      #and the pairwise D_e for each distribution and pair of iterations
      expectations <- estimate_shuffled_single_expectation(joint_alpha,
                                                           pairwise_alpha,
                                                           pairwise, samples,
                                                           lower_bound, method,
                                                           length(x), zero2,
                                                           verbose)
   } else {
      #Use the standard formula to compute the joint D_e for each distribution
      #and the pairwise D_e for each distribution and pair of iterations
      expectations <- compute_standard_single_expectation(joint_alpha,
                                                          pairwise_alpha,
                                                          pairwise, verbose)
   }
   de_single_joint <- expectations[[1]]
   if(pairwise) {
      de_single_pairwise <- expectations[[2]]
   }

   reliability <- vector("list",4)
   #Will contain up to four types of reliability coefficients:
   #1. single-distribution joint reliability
   #2. single-distribution pairwise reliability
   #3. omnibus joint reliability
   #4. omnibus pairwise reliability

   reliability[[1]] <- (1 - (do_single_joint / de_single_joint))
   reliability[[3]] <- (1 - (sum(do_single_joint) / sum(de_single_joint)))
   if(pairwise) {
      reliability[[2]] <- do_single_pairwise
      reliability[[4]] <- do_single_pairwise
      #The above lines just set reliability[[2]] and reliability[[4]] to the
      #correct dimensions; all values copied from do_single_pairwise will be
      #overwritten
      for(i in 1:(length(x)-1)) { #For each iteration i...
         for(j in 1:length(x)) { #And for each fellow iteration j...
            if(verbose) {
               message(paste("Preparing pairwise reliability output:",
                             "Iterations", i, "and", j, "of", length(x)))
               utils::flush.console()
            }
            reliability[[2]][[i]][[j]] <- (1 - (do_single_pairwise[[i]][[j]] /
                                                de_single_pairwise[[i]][[j]]))
            reliability[[4]][[i]][[j]] <-
                                       (1 - (sum(do_single_pairwise[[i]][[j]]) /
                                             sum(de_single_pairwise[[i]][[j]])))
         }
      }
   }
   return(reliability)
}

#' Generate Squared Dirichlet Deviate Differences
#'
#' @description This helper function generates two deviates from the same
#'   underlying Dirichlet distribution and returns the sum of the squared
#'   differences between the categories of those deviates.
#'
#' @param x The concentration parameters used to generate Dirichlet deviates
#' @param samples An integer value indicating how many samples to use to
#'   compute the sum of squared differences
#' @param zero2 A numeric value; in order to prevent errors when generating
#'   Dirichlet deviates, any concentration parameters that are less than
#'   \code{zero2} are changed to be \code{zero2}
#' @return A numeric value representing the sum of squared differences between
#'   Dirichlet deviates for \code{samples} times the number of pairs of deviates
#' @param method A string indicating what method (\code{"rotational"} or
#'   \code{"forward"}) will be used to reorder topics (see Britt, under review);
#'   the \code{"forward"} method will be implemented at a later date
#' @export
#' generate_dirichlet_deviate_diffs

generate_dirichlet_deviate_diffs <- function(x, samples, zero2=(10^(-255))) {
   return(sum((rdirichlet(x, samples, zero2)-rdirichlet(x, samples, zero2))^2))
}

#' Estimate Shuffled Single Expectation
#'
#' @description This helper function estimates the expected squared difference
#'   between deviates from the same underlying Dirichlet distribution when the
#'   categories in each distribution are allowed to be reordered.
#'
#' @param joint_alpha A vector of concentration parameters representing all
#'   cross-validation iterations
#' @param pairwise_alpha A list of lists of vectors of concentration parameters
#'   representing all pairs of cross-validation iterations
#' @param pairwise A boolean value indicating whether pairwise reliability
#'   between individual cross-validation iterations should be computed; if
#'   \code{FALSE}, pairwise reliability will be ignored in order to reduce the
#'   time and memory complexity of the computation
#' @param samples An integer value indicating how many samples to use to
#'   estimate expected differences when \code{shuffle=TRUE}
#' @param lower_bound A boolean value indicating whether the lower bound of the
#'   expected differences (based on i.i.d. beta distributions) should be used
#'   rather than shuffling Dirichlet distributions to estimate the exact
#'   difference in every sample when \code{shuffle=TRUE}; for Dirichlet
#'   distributions with many categories and function calls with a large value of
#'   samples, setting this to \code{TRUE} is strongly recommended, as reordering
#'   numerous samples with a large number of categories may require an
#'   unreasonable amount of time
#' @param method A string indicating what method (\code{"rotational"} or
#'   \code{"forward"}) will be used to reorder topics (see Britt, under review);
#'   the \code{"forward"} method will be implemented at a later date
#' @param num_iter An integer value indicating the number of cross-validation
#'   iterations
#' @param zero2 A numeric value; in order to prevent errors when generating
#'   Dirichlet deviates, any concentration parameters that are less than
#'   \code{zero2} are changed to be \code{zero2}
#' @param verbose A boolean value indicating whether progress updates should be
#'   provided
#' @return A two-element list indicating the expected difference between
#'   reordered Dirichlet deviates based on the concentration parameters given in
#'   \code{joint_alpha} and \code{pairwise_alpha}, respectively
#' @export
#' estimate_shuffled_single_expectation

estimate_shuffled_single_expectation <- function(joint_alpha, pairwise_alpha,
                                            pairwise, samples=1000,
                                            lower_bound=TRUE,
                                            method="rotational",
                                            num_iter=(length(pairwise_alpha)+1),
                                            zero2=(10^(-255)), verbose=FALSE) {
   #Expected difference for one distribution across all iterations
   single_joint_expectation <- rep(0,length(joint_alpha))
   #Expected difference for one distribution across pairs of iterations
   if(pairwise) {
      single_pairwise_expectation <- vector("list",length(pairwise_alpha))
      for(i in 1:length(single_pairwise_expectation)) {
         if(verbose) {
            message(paste("Initial setting of 'single_pairwise_expectation':",
                          "Step", i, "of", length(single_pairwise_expectation)))
            utils::flush.console()
         }
         single_pairwise_expectation[[i]] <- vector("list",
                                                    length(pairwise_alpha[[i]]))
      }
   } else {
      single_pairwise_expectation = NA
   }

   if(lower_bound) { #Estimate the lower bound of the expected difference
      for(i in 1:length(joint_alpha)) {
         if(verbose) {
            message(paste("Computing 'single_joint_expectation': Step", i, "of",
                          length(joint_alpha)))
            utils::flush.console()
         }
         single_joint_expectation[i] <- sum((rdirichlet(joint_alpha[[i]],
                                                        samples, zero2) - 
                                             rdirichlet(joint_alpha[[i]],
                                                        samples, zero2))^2)
      }
      single_joint_expectation <- single_joint_expectation / samples
      if(pairwise) {
         for(i in 1:(length(pairwise_alpha[[1]])-1)) {
            for(j in (i+1):length(pairwise_alpha[[i]])) {
               if(verbose) {
                  message(paste("Setting up 'single_pairwise_expectation':",
                                "Iterations", i, "and", j, "of",
                                length(pairwise_alpha[[i]])))
                  utils::flush.console()
               }
               single_pairwise_expectation[[i]][[j]] <-
                                           rep(0,nrow(pairwise_alpha[[i]][[j]]))
            }
         }
         for(i in 1:(length(pairwise_alpha[[1]])-1)) {
            for(j in (i+1):length(pairwise_alpha[[i]])) {
               if(verbose) {
                  message(paste("Computing 'single_pairwise_expectation':",
                                "Iterations", i, "and", j, "of",
                                length(pairwise_alpha[[i]])))
                  utils::flush.console()
               }
               single_pairwise_expectation[[i]][[j]] <-
                                            apply(pairwise_alpha[[i]][[j]],1,
                                               generate_dirichlet_deviate_diffs,
                                               samples=samples, zero2=zero2)
            }
         }
         for(i in 1:(length(pairwise_alpha[[1]])-1)) {
            for(j in (i+1):length(pairwise_alpha[[i]])) {
               if(verbose) {
                  message(paste("Adjusting 'single_pairwise_expectation':",
                                "Iterations", i, "and", j, "of",
                                length(pairwise_alpha[[i]])))
                  utils::flush.console()
               }
               single_pairwise_expectation[[i]][[j]] <-
                                 single_pairwise_expectation[[i]][[j]] / samples
            }
         }
      }
   } else {
      #Estimate the exact expected difference; may be computationally intensive
      for(i in 1:samples) {
         joint_data <- vector("list",num_iter)
         for(j in 1:num_iter) {
            joint_data[[j]] <- rdirichlet(joint_alpha[[1]], 1, zero2)
            if(verbose) {
               message(paste("Preparing 'joint_alpha': Sample " , i, " of ",
                             samples, ", Category ", j, " of " , num_iter,
                             sep=""))
               utils::flush.console()
            }
            for(k in 2:length(joint_alpha)) {
               joint_data[[j]] <- rbind(joint_data[[j]],
                                        rdirichlet(joint_alpha[[k]], 1, zero2))
            }
         }
         shuffled_joint_data <- shuffle_distributions(joint_data, method,
                                                      verbose)[[1]]
         for(j in 1:(length(shuffled_joint_data)-1)) {
            for(k in (j+1):length(shuffled_joint_data)) {
               if(verbose) {
                  message(paste("Computing 'single_joint_expectation': ",
                             "Sample ", i, " of ", samples, ", Distributions ",
                             j, " and ", k, " of " ,
                             length(shuffled_joint_data), sep=""))
                  utils::flush.console()
               }
               single_joint_expectation <- single_joint_expectation +
                  rowSums((shuffled_joint_data[[j]]-shuffled_joint_data[[k]])^2)
            }
         }
      }
      single_joint_expectation <- single_joint_expectation*2 /
                                  (samples*num_iter*(num_iter-1))
      if(pairwise) {
         for(i in 1:(length(single_pairwise_expectation)-1)) {
            #Since new data must be created, the pairwise expectation uses the
            #joint alphas
            for(j in (i+1):length(single_pairwise_expectation)) {
               if(verbose) {
                  message(paste("Computing 'single_pairwise_expectation':",
                                "Iterations", i, "and", j, "of",
                                length(single_pairwise_expectation)))
                  utils::flush.console()
               }
               single_pairwise_expectation[[i]][[j]] <- single_joint_expectation
            }
         }
      }
   }
   return(list(single_joint_expectation, single_pairwise_expectation))
}

#' Compute Standard Single Expectation
#'
#' @description This helper function computes the expected squared difference
#'   between deviates from the same underlying Dirichlet distribution when the
#'   categories in each distribution are not allowed to be reordered.
#'
#' @param joint_alpha A vector of concentration parameters representing all
#'   cross-validation iterations
#' @param pairwise_alpha A list of lists of vectors of concentration parameters
#'   representing all pairs of cross-validation iterations
#' @param pairwise A boolean value indicating whether pairwise reliability
#'   between individual cross-validation iterations should be computed; if
#'   \code{FALSE}, pairwise reliability will be ignored in order to reduce the
#'   time and memory complexity of the computation
#' @param verbose A boolean value indicating whether progress updates should be
#'   provided
#' @return A two-element list indicating the expected difference between
#'   reordered Dirichlet deviates based on the concentration parameters given in
#'   \code{joint_alpha} and \code{pairwise_alpha}, respectively
#' @export
#' compute_standard_single_expectation

compute_standard_single_expectation <- function(joint_alpha, pairwise_alpha,
                                                pairwise, verbose=FALSE) {
   de_individual_joint <- vector("double")
   sum_de_individual_joint <- vector("double")
   if(pairwise) {
      de_individual_pairwise <- vector("list",(length(pairwise_alpha)))
      for(i in 1:length(de_individual_pairwise)) {
         de_individual_pairwise[[i]] <- vector("list",
                                               length(pairwise_alpha[[i]]))
         for(j in (i+1):length(de_individual_pairwise[[i]])) {
            if(verbose) {
               message(paste("Setting up 'de_individual_pairwise': Iterations",
                             i, "and", j, "of",
                             length(de_individual_pairwise[[i]])))
               utils::flush.console()
            }
            de_individual_pairwise[[i]][[j]] <- vector("double")
         }
      }
      sum_de_individual_pairwise <- de_individual_pairwise
   } else {
      de_individual_pairwise <- NA
   }

   for(i in 1:length(joint_alpha)) { #Compute D_e_joint for each distribution
      if(verbose) {
         message(paste("Computing 'de_individual_joint': Category", i, "of",
                       length(joint_alpha)))
         utils::flush.console()
      }
      de_individual_joint[i] <- 0
      sum_de_individual_joint[i] <- 0
      for(j in 1:length(joint_alpha[[i]])) { #Compute a_0
         sum_de_individual_joint[i] <- sum_de_individual_joint[i] +
                                       joint_alpha[[i]][j]
      }
      for(j in 1:length(joint_alpha[[i]])) { #Compute summation for D_e_joint
         de_individual_joint[i] <- de_individual_joint[i] +
                                   (((2*joint_alpha[[i]][j]) *
                                     (sum_de_individual_joint[i] -
                                      joint_alpha[[i]][j])) /
                                    ((sum_de_individual_joint[i]^2) *
                                     (sum_de_individual_joint[i]+1)))
      }
   }

   if(pairwise) {
      for(i in 1:(length(pairwise_alpha[[1]])-1)) { #For each iteration i...
         for(j in (i+1):length(pairwise_alpha[[i]])) {
            #And for each fellow iteration j...
            if(verbose) {
               message(paste("Computing 'de_individual_pairwise': Iterations",
                             i, "and", j, "of", length(pairwise_alpha)))
               utils::flush.console()
            }
            for(k in 1:nrow(pairwise_alpha[[i]][[j]])) {
               #Compute D_e_pairwise for each distribution
               de_individual_pairwise[[i]][[j]][k] <- 0
               sum_de_individual_pairwise[[i]][[j]][k] <- 0
               for(l in 1:length(pairwise_alpha[[i]][[j]][k,])) { #Compute a_0
                  sum_de_individual_pairwise[[i]][[j]][k] <-
                     sum_de_individual_pairwise[[i]][[j]][k] +
                     pairwise_alpha[[i]][[j]][k,l]
               }
               for(l in 1:length(pairwise_alpha[[i]][[j]][k,])) {
                  #Compute summation for D_e_pairwise
                  de_individual_pairwise[[i]][[j]][k] <-
                     de_individual_pairwise[[i]][[j]][k] +
                     (((2*pairwise_alpha[[i]][[j]][k,l]) *
                       (sum_de_individual_pairwise[[i]][[j]][k] - 
                        pairwise_alpha[[i]][[j]][k,l])) /
                      ((sum_de_individual_pairwise[[i]][[j]][k]^2) *
                       (sum_de_individual_pairwise[[i]][[j]][k]+1)))
               }
            }
         }
      }
   }
   return(list(de_individual_joint, de_individual_pairwise))
}

#' Estimate Alpha from Data
#'
#' @description This helper function estimates the expected concentration
#'   parameters of the Dirichlet distribution underlying multiple deviates.
#'
#' @param x A list with each entry being a single cross-validation iteration;
#'   each list entry should be either the output from a
#'   \code{\link[topicmodels]{LDA}} function call or a 2D double vector with
#'   distributions as the rows and each category within each distribution as the
#'   columns (such that each row in the vector sums to 1)
#' @param pairwise A boolean value indicating whether pairwise reliability
#'   between individual cross-validation iterations should be computed; if
#'   \code{FALSE}, pairwise reliability will be ignored in order to reduce the
#'   time and memory complexity of the computation
#' @param estimate_pairwise_alpha_from_joint A boolean value indicating, when
#'   estimating reliability for pairs of cross-validation iterations, whether
#'   each pair of iterations should use the concentration parameters estimated
#'   across all iterations (\code{TRUE}) or whether separate sets of
#'   concentration parameters should be estimated for each pair of
#'   cross-validation iterations (\code{FALSE}); it is strongly suggested that
#'   the default value of \code{TRUE} be used for this parameter
#' @param symmetric_alpha A boolean value indicating whether all concentration
#'   parameters in each cross-validation iteration should be assumed to be
#'   equal, which is common in LDA
#' @param convcrit A numeric value indicating the threshold for convergence used
#'   to estimate concentration parameters
#' @param maxit A numeric value indicating the maximum number of iterations used
#'   to estimate concentration parameters
#' @param verbose A boolean value indicating whether progress updates should be
#'   provided
#' @return A two-element list indicating the estimated concentration parameters
#'   for all concentration parameters and for each pair of cross-validation
#'   iterations, respectively
#' @export
#' estimate_alpha_from_data

estimate_alpha_from_data <- function(x, pairwise,
                                     estimate_pairwise_alpha_from_joint,
                                     symmetric_alpha=FALSE, convcrit=0.00001,
                                     maxit=1000, verbose=FALSE) {
   data <- vector("list",nrow(x[[1]]))
   joint_alpha <- data
   pairwise_alpha <- NA
   for(i in 1:nrow(x[[1]])) {
      #Extract a single distribution across multiple iterations
      if(verbose) {
         message(paste("Extracting distribution", i, "of", nrow(x[[1]])))
         utils::flush.console()
      }
      data[[i]] <- x[[1]][i,]
      for(j in 2:length(x)) {
         data[[i]] <- rbind(data[[i]], x[[j]][i,])
      }
   }
   #Create list with one element for each distribution, each of which is a
   #vector of alphas
   for(i in 1:length(data)) {
      if(verbose) {
         message(paste("Setting up 'joint_alpha': Step", i, "of",
                       length(data)))
         utils::flush.console()
      }
      if(symmetric_alpha) { #Equally partition alpha0 among all categories
         joint_alpha[[i]] <- rep((symmetric.dirichlet.mle(data[[i]],
                                                     convcrit=convcrit,
                                                     maxit=maxit,
                                                     verbose=verbose)$`alpha0` /
                                  length(data[[i]])),length(data[[i]]))
      } else {
         #Use alpha as-is, potentially with different values for each category
         joint_alpha[[i]] <- dirichlet.mle(data[[i]], convcrit=convcrit,
                                           maxit=maxit,
                                           verbose=verbose)$`alpha`
      }
   }
   if(pairwise) {
      #Create nested list: top level is iterations, second level is iterations,
      #third level is distributions, each of which is a vector of alphas
      pairwise_alpha <- vector("list",(length(x)-1))
      for(i in 1:length(x)) {
         if(verbose) {
            message(paste("Setting up 'pairwise_alpha': Step", i, "of",
                          length(x)))
            utils::flush.console()
         }
         pairwise_alpha[[i]] <- vector("list",length(x))
      }
      if(estimate_pairwise_alpha_from_joint) {
         for(i in 1:(length(x)-1)) {
            for(j in (i+1):length(x)) {
               if(verbose) {
                  message(paste("Adding data to 'pairwise_alpha': Iterations",
                                i, "and", j, "of", length(x)))
                  utils::flush.console()
               }
               pairwise_alpha[[i]][[j]] <- joint_alpha[[1]]
               for(k in 2:nrow(x[[i]])) {
                  pairwise_alpha[[i]][[j]] <- rbind(pairwise_alpha[[i]][[j]],
                                                    joint_alpha[[k]])
               }
            }
         }
      } else {
         for(i in 1:(length(x)-1)) {
            for(j in (i+1):length(x)) {
               if(verbose) {
                  message(paste("Computing 'pairwise_alpha': Iterations ", i,
                                " and ", j, " of ", length(x),
                                ", Category 1 of ", nrow(x[[i]]), sep=""))
                  utils::flush.console()
               }
               if(symmetric_alpha) { #Equally partition alpha0 in all categories
                  pairwise_alpha[[i]][[j]] <-
                     rep((symmetric.dirichlet.mle(rbind(x[[i]][1,],
                                                        x[[j]][1,]),
                                                  convcrit=convcrit,
                                                  maxit=maxit,
                                                  verbose=verbose)$`alpha0` /
                          length(rbind(x[[i]][1,],x[[j]][1,]))),
                         length(rbind(x[[i]][1,],x[[j]][1,])))
               } else {
                  pairwise_alpha[[i]][[j]] <- dirichlet.mle(rbind(x[[i]][1,],
                                                                 x[[j]][1,]),
                                                        convcrit=convcrit,
                                                        maxit=maxit,
                                                        verbose=verbose)$`alpha`
               }
               for(k in 2:nrow(x[[i]])) {
                  if(verbose) {
                     message(paste("Computing 'pairwise_alpha': Iterations ", i,
                                   " and ", j, " of ", length(x), ", Category ",
                                   k, " of ", nrow(x[[i]]), sep=""))
                     utils::flush.console()
                  }
                  if(symmetric_alpha) {
                     #Equally partition alpha0 among all categories
                     pairwise_alpha[[i]][[j]] <- rbind(pairwise_alpha[[i]][[j]],
                      rep((symmetric.dirichlet.mle(rbind(x[[i]][k,],x[[j]][k,]),
                                                   convcrit=convcrit,
                                                   maxit=maxit,
                                                   verbose=verbose)$`alpha0` /
                           length(rbind(x[[i]][k,],x[[j]][k,]))),
                          length(rbind(x[[i]][k,],x[[j]][k,]))))
                  } else {
                     #Use alpha as-is, potentially with different values for
                     #each category
                     pairwise_alpha[[i]][[j]] <- 
                        rbind(pairwise_alpha[[i]][[j]],
                              dirichlet.mle(rbind(x[[i]][k,],x[[j]][k,]),
                                            convcrit=convcrit, maxit=maxit,
                                            verbose=verbose)$`alpha`)
                  }
               }
            }
         }
      }
   }
   return(list(joint_alpha,pairwise_alpha))
}

#' Shuffle Distributions
#'
#' @description This helper function reorders the categories in multiple
#'   Dirichlet deviates in order to minimize the sum of squared differences
#'   between them.
#'
#' @param x A list with each entry being a single cross-validation iteration;
#'   each list entry should be either the output from a
#'   \code{\link[topicmodels]{LDA}} function call or a 2D double vector with
#'   distributions as the rows and each category within each distribution as the
#'   columns (such that each row in the vector sums to 1)
#' @param method A string indicating what method (\code{"rotational"} or
#'   \code{"forward"}) will be used to reorder topics (see Britt, under review);
#'   the \code{"forward"} method will be implemented at a later date
#' @param verbose A boolean value indicating whether progress updates should be
#'   provided
#' @return A two-element list containing the reordered data set and a matrix
#'   indicating the relationship between each reordered category and its orginal
#'   position in the data
#' @export
#' shuffle_distributions

shuffle_distributions <- function(x, method="rotational", verbose=FALSE) {
   step <- 0 #Used to track progress when verbose=TRUE
   c <- length(x) #Number of iterations
   k <- nrow(x[[1]]) #Number of distributions (e.g., topics)
   v <- ncol(x[[1]]) #Number of categories in each distribution (e.g., words)
   assignments <- t(matrix(1:k, k, c))
   alldiff <- array(data = NA, dim = c(c,c,k,k))

   squarediff <- function(x,y) sum((x-y)^2)
   improve_increment <- function(v,w,x,y,improvement) {
      return(improvement+v+w-x-y)
   }
   set_change <- function(p,q,r,s,improvement,change,iter,dist1,dist2) {
      if(improvement > change) {
         #This is the best improvement detected so far...
         change <- improvement
         iter <- p
         dist1 <- r
         dist2 <- s
      }
      return(list(change,iter,dist1,dist2))
   }
   set_change2 <- function(alldiff,p,r,s,improvement,change,iter,dist1,dist2) {
      for(q in (p+1):c) {
         improvement <- improve_increment(alldiff[p,q,r,r],
                                          alldiff[p,q,s,s],
                                          alldiff[p,q,r,s],
                                          alldiff[p,q,s,r],improvement)
         changelist <- set_change(p,q,r,s,improvement,change,iter,dist1,dist2)
      }
      return(changelist)
   }
   set_change3 <- function(alldiff,p,r,improvement,change,iter,dist1,dist2) {
      temp <- sapply((r+1):k, function(s) set_change2(alldiff,p,r,s,improvement,
                                                      change,iter,dist1,dist2))
      return(temp[,which.max(temp[1,])])
   }
   set_change4 <- function(alldiff,p,improvement,change,iter,dist1,dist2) {
      temp <- sapply(1:(k-1), function(r) set_change3(alldiff,p,r,improvement,
                                                      change,iter,dist1,dist2))
      return(temp[,which.max(temp[1,])])
   }
   set_change5 <- function(alldiff,improvement,change,iter,dist1,dist2) {
      temp <- sapply(1:(c-1), function(p) set_change4(alldiff,p,improvement,
                                                      change,iter,dist1,dist2))
      return(temp[,which.max(temp[1,])])
   }
   swap <- function(p,r,iter,dist1,dist2,alldiff) {
      temp_double <- alldiff[p,iter,r,dist1]
      alldiff[p,iter,r,dist1] <- alldiff[p,iter,r,dist2]
      alldiff[p,iter,r,dist2] <- temp_double
      temp_double <- alldiff[iter,p,dist1,r]
      alldiff[iter,p,dist1,r] <- alldiff[iter,p,dist2,r]
      alldiff[iter,p,dist2,r] <- temp_double
      return(alldiff)
   }
   reorder_assignments <- function(x,y,p,r,assignments) {
      y[[p]][r,] <- x[[p]][assignments[p,r],]
   }

   for(p in 1:(c-1)) {
      for(q in (p+1):c) {
         if(verbose) {
            message(paste("Computing sum of squared differences: Distributions",
                          p, "and", q, "of", c))
            utils::flush.console()
         }
         alldiff[p,q,,] <- sapply(1:k, function(s) sapply(1:k,
                                function(r) squarediff(x[[p]][r,], x[[q]][s,])))
      }
   }

   if(method=="forward") {
      #NOT YET IMPLEMENTED
   }

   if(method=="rotational") {
      #Like a Rubik's cube, this approach swaps pairs of distributions within a
      #given iteration, one at a time, until the overall fit can no longer be
      #improved with any additional swap
      step <- 0
      #Reduce d_o by repeatedly switching two distributions in one iteration
      repeat {
         step <- step + 1
         if(verbose) {
            message(paste("Distribution switching: Step", step))
            utils::flush.console()
         }
         change = 0
         improvement <- 0
         iter = 0
         dist1 = 0
         dist2 = 0

         changelist <- set_change5(alldiff,improvement,change,iter,dist1,dist2)
         change <- changelist[[1]]
         iter <- changelist[[2]]
         dist1 <- changelist[[3]]
         dist2 <- changelist[[4]]
         if(change > 0) {
            #If at least one distribution switch would reduce observed diff...
            #Make the most productive switch in the assignments matrix
            temp_int = assignments[iter,dist1]
            assignments[iter,dist1] = assignments[iter,dist2]
            assignments[iter,dist2] = temp_int
            #Update the alldiff matrix
            for(p in 1:c) {
               for(r in 1:k) {
                  if(verbose) {
                     message(paste("Updating the 'alldiff' matrix: Step ", step,
                                   ", Distributions ", p, " and ", r, sep=""))
                     utils::flush.console()
                  }
                  temp_double <- alldiff[p,iter,r,dist1]
                  alldiff[p,iter,r,dist1] <- alldiff[p,iter,r,dist2]
                  alldiff[p,iter,r,dist2] <- temp_double
                  temp_double <- alldiff[iter,p,dist1,r]
                  alldiff[iter,p,dist1,r] <- alldiff[iter,p,dist2,r]
                  alldiff[iter,p,dist2,r] <- temp_double
               }
            }
         } else {
            break
         }
      }

      #Reorder the distributions in accordance with the assignments matrix
      y <- x
      for(p in 1:c) {
         for(r in 1:k) {
            if(verbose) {
               message(paste("Reorder distributions based on 'assignments'",
                             "matrix: Distributions", p, "and", r))
               y[[p]][r,] <- x[[p]][assignments[p,r],]
            }
         }
      }
      return(list(y, assignments))
   }
}

#' Dirichlet.MLE
#'
#' @description This helper function estimates the concentration parameters of
#'   the Dirichlet distribution underlying multiple deviates, with no
#'   restrictions placed on the values of those concentration parameters. This
#'   function is heavily based on \code{\link[sirt]{dirichlet.mle}}, with
#'   modifications to avoid potential singularities in the estimation procedure.
#'
#' @param x A list with each entry being a single cross-validation iteration;
#'   each list entry should be either the output from a
#'   \code{\link[topicmodels]{LDA}} function call or a 2D double vector with
#'   distributions as the rows and each category within each distribution as the
#'   columns (such that each row in the vector sums to 1)
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
#' @param verbose A boolean value indicating whether progress updates should be
#'   provided
#' @return A list of the estimated concentration parameters, the sum of those
#'   estimated concentration parameters, and the ratio between each estimated
#'   concentration parameter and the sum of those parameters
#' @export
#' dirichlet.mle

dirichlet.mle <- function(x, weights=NULL, eps=10^(-5), convcrit=.00001,
        maxit=1000, oldfac=.3, verbose=FALSE)
{
   #Taken from https://rdrr.io/cran/sirt/src/R/dirichlet.mle.R, but with minor
   #modifications (especially in the initial setting of xsi) to avoid potential
   #singularities in the estimation procedure.
   N <- nrow(x)
   K <- ncol(x)
   # compute log pbar
   x <- (x+eps)/(1+2*eps)
   x <- x/rowSums(x)
   N <- nrow(x)
   if (is.null(weights)){
      weights <- rep(1,(N*K))
   }
   weights <- N*weights/sum(weights)
   log.pbar <- colMeans(weights*log(x))
   # compute inits
   alphaprob <- colMeans(x*weights)
   p2 <- mean( unlist(x)^2*weights )
   #The original version of the above line, as it appears in the sirt package:
   #p2 <- mean( x[,1]^2 * weights )
   xsi <- (mean(alphaprob)-p2)/(p2-(mean(alphaprob))^2)
   #The original version of the above line, as it appears in the sirt package:
   #xsi <- ( alphaprob[1] - p2 ) / ( p2 - ( alphaprob[1] )^2 )
   alpha <- xsi*alphaprob
   K1 <- matrix(1,K,K)
   conv <- 1
   iter <- 1
   #The below line was not in the original function as it appeared in the sirt
   #package
   alpha[alpha < (1e-10)] <- 1e-10

   #--- BEGIN iterations
   while( ( conv > convcrit ) & (iter < maxit) ){
      if(verbose){
         message(paste("Estimating Dirichlet concentration parameters: Step ",
                       iter, " of ", maxit, ", 'conv' = ", conv, sep=""))
         utils::flush.console()
      }
      alpha0 <- alpha
      g <- N * K * digamma(sum(alpha)) -
           N * K * digamma(alpha) + N * K * log.pbar
      z <- N * K * sirt_digamma1(sum(alpha))
      H <- diag( -N * K * sirt_digamma1( alpha ) ) + z
      #Original versions of these lines from the sirt package
      #g <- N * digamma( sum(alpha ) ) - N * digamma(alpha) + N * log.pbar
      #z <- N * sirt_digamma1( sum(alpha ))
      #H <- diag( -N * sirt_digamma1( alpha ) ) + z
      alpha <- alpha0 - solve(H, g )
      alpha[ alpha < (1e-10) ] <- 1e-10
      #Original version of the above line, as the function appeared in the sirt
      #package
      #alpha[ alpha < 0 ] <- 1e-10 
      alpha <- alpha0 + oldfac*(alpha-alpha0 )
      conv <- max(abs(alpha0-alpha))
      iter <- iter+1
   }
   alpha0 <- sum(alpha)
   xsi <- alpha / alpha0
   res <- list( alpha=alpha, alpha0=alpha0, xsi=xsi )
   return(res)
}

#' Symmetric.Dirichlet.MLE
#'
#' @description This helper function estimates the concentration parameters of
#'   the Dirichlet distribution underlying multiple deviates, assuming that
#'   those concentration parameters are all equal. This function is heavily
#'   based on \code{\link[sirt]{dirichlet.mle}}, with modifications to restrict
#'   all concentration parameters to be equal and to avoid potential
#'   singularities in the estimation procedure.
#'
#' @param x A list with each entry being a single cross-validation iteration;
#'   each list entry should be either the output from a
#'   \code{\link[topicmodels]{LDA}} function call or a 2D double vector with
#'   distributions as the rows and each category within each distribution as the
#'   columns (such that each row in the vector sums to 1)
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
#' @param verbose A boolean value indicating whether progress updates should be
#'   provided
#' @return A list of the estimated concentration parameters, the sum of those
#'   estimated concentration parameters, and the ratio between each estimated
#'   concentration parameter and the sum of those parameters
#' @export
#' symmetric.dirichlet.mle

symmetric.dirichlet.mle <- function(x, weights=NULL, eps=10^(-5),
                                    convcrit=.00001, maxit=1000, oldfac=.3,
                                    verbose=FALSE)
#Taken from https://rdrr.io/cran/sirt/src/R/dirichlet.mle.R, but with minor
#modifications (especially in the initial setting of xsi) to avoid potential
#singularities in the estimation procedure.
{
   # compute log pbar
   x <- ( x+eps ) / ( 1 + 2*eps )
   x <- x / rowSums(x)
   N <- nrow(x)
   K <- ncol(x)
   if ( is.null(weights) ){
      #weights <- rep(1,N)
      weights <- rep(1,(N*K))
   }
   weights <- N * weights / sum( weights )
   log.pbar <- colMeans( weights * log( x ) )
   # compute inits
   alphaprob <- colMeans( x * weights )
   p2 <- mean( unlist(x)^2 * weights )
   #Original version of the above line, as the function appeared in the sirt
   #package
   #p2 <- mean( x[,1]^2 * weights )
   xsi <- ( mean(alphaprob) - p2 ) / ( p2 - ( mean(alphaprob) )^2 )
   #Original version of the above line, as the function appeared in the sirt
   #package
   #xsi <- ( alphaprob[1] - p2 ) / ( p2 - ( alphaprob[1] )^2 )
   alpha <- xsi * alphaprob
   K1 <- matrix(1,K,K)
   conv <- 1
   iter <- 1
   #The below line was not in the original function as it appeared in the sirt
   #package
   alpha[alpha < (1e-10)] <- 1e-10 
   alpha0 <- rep(mean(alpha),length(alpha))
   alpha <- alpha0

   #--- BEGIN iterations
   while( ( conv > convcrit ) & (iter < maxit) ){
      if(verbose) {
         message(paste("Estimating Dirichlet concentration parameters: Step ",
                       iter, " of ", maxit, ", 'conv' = ", conv, sep=""))
         utils::flush.console()
      }
      #Given a matrix defining of a system of equations, where the diagonal
      #elements = a and the off-diagonal elements = b, and the resulting vector
      #is named g, the solution vector may be expressed as:
      #solution[i] = -g[i]*(a+(length(g)-2)*b)/((a+(length(g)-1)*b)*(b-a)) -
      #              g[i]*(b)/((a+(length(g)-1)*b)*(b-a)) +
      #              sum(g*b/((a+(length(g)-1)*b)*(b-a)))
      #In this case, the diagonal elements =
      #  (-N*K*(digamma(alpha+(1e-3)) - digamma(alpha-(1e-3)))/(2*(1e-3)))[1]+z
      #and the off-diagonal elements = z. (Refer to dirichlet.mle().)
      #Thus, rather than using a computationally intensive system of equations
      #for a square matrix with K rows and columns to run solve(H,g), as is done
      #in dirichlet.mle() to account for non-symmetric matrices, we can simply
      #use the following for symmetric matrices:
      g <- N * digamma( sum(alpha ) ) - N * digamma(alpha) + N * log.pbar
      b <- N * sirt_digamma1( sum(alpha ))
      a <- (-N*(digamma(alpha+(1e-3))-digamma(alpha-(1e-3)))/(2*(1e-3)))[1]+b
      solution <- numeric(0)
      for(i in 1:length(g)) {
         solution[i] <- -g[i]*(a+(length(g)-2)*b)/((a+(length(g)-1)*b)*(b-a)) -
                        g[i]*(b)/((a+(length(g)-1)*b)*(b-a)) +
                        sum(g*b/((a+(length(g)-1)*b)*(b-a)))
      }
      alpha <- alpha0 - solution
      alpha[alpha < (1e-10)] <- 1e-10
      #Original version of the above line, as the function appeared in the sirt
      #package
      #alpha[ alpha < 0 ] <- 1e-10
      #Set all concentration parameters to be equal rather than allowing them
      #to freely vary
      alpha <- rep(mean(alpha),length(alpha))
      conv <- abs( alpha0[1] - alpha[1] )
      alpha <- alpha0 + oldfac*(alpha - alpha0)
      alpha0 <- alpha
      iter <- iter+1
   }
   alpha0 <- sum(alpha)
   xsi <- alpha / alpha0
   res <- list(alpha=alpha, alpha0=alpha0, xsi=xsi)
   return(res)
}

#' Sirt Digamma 1
#'
#' @description This helper function estimates the derivative of the digamma
#'   function. This function is directly drawn from
#'   \code{\link[sirt]{sirt_digamma1}}, as published at
#'   \url{https://github.com/cran/sirt/blob/master/R/sirt_digamma1.R}.
#'
#' @param x A vector of concentration parameters
#' @param h A numeric value used to define a pair of nearby observations
#' @return A list of the estimated concentration parameters, the sum of those
#'   estimated concentration parameters, and the ratio between each estimated
#'   concentration parameter and the sum of those parameters
#' @export
#' symmetric.dirichlet.mle

sirt_digamma1 <- function(x, h=1e-3)
#Taken from https://rdrr.io/cran/sirt/src/R/sirt_digamma1.R
{
   ( digamma(x+h) - digamma(x-h) ) / (2*h)
}

#' RDirichlet
#'
#' @description This helper function generates a set of deviates from a
#'   Dirichlet distribution with underlying concentration parameters equal to
#'   \code{alpha}. This function is heavily based on
#'   \code{\link[sirt]{rdirichlet}}, with modifications to prevent concentration
#'   parameters equal to 0.
#'
#' @param alpha A numeric vector indicating the concentration parameters of the
#'   Dirichlet distribution from which deviates should be generated
#' @param n A numeric value indicating the number of Dirichlet deviates to be
#'   generated
#' @param zero2 A numeric value representing the minimum allocation permitted
#'   for any given Dirichlet category in order to prevent errors
#' @return A numeric vector with \code{n} Dirichlet deviates
#' @export
#' symmetric.dirichlet.mle

rdirichlet <- function(alpha, n, zero2=(10^(-255))) #Based on sirt::rdirichlet
## generate n random deviates from the Dirichlet function with shape
## parameters alpha
{
    l<-length(alpha)
    x<-matrix(stats::rgamma(l*n,alpha),ncol=l,byrow=TRUE)
    sm<-as.vector(x%*%rep(1,l)) #Modified from the original function in sirt
    #The below line was not in the original function in sirt; it changes sm
    #values from 0 to "zero2" as needed
    sm[sm < (zero2)] <- zero2
    values <- x/as.vector(sm)
    return(values)
}