% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/brittnu.R
\name{print.euclidkrip_rel}
\alias{print.euclidkrip_rel}
\title{Print (euclidkrip)}
\usage{
\method{print}{euclidkrip_rel}(x, element = NA, ...)
}
\arguments{
\item{x}{An object of class \code{euclidkrip_rel} outputted from
\link{euclidkrip}}

\item{element}{The specific reliability coefficient to be printed; permitted
values are \code{"singleomnibus"}, \code{"singlepairwise"},
\code{"multipleomnibus"}, \code{"multiplepairwise"}, \code{"all"}, and
\code{NA}}

\item{...}{Other arguments inherited from the generic \code{print} function}
}
\description{
This helper method calls \code{summary()} when the user attempts
  to print an object of class \code{euclidkrip_rel} outputted from
  \link{euclidkrip}.
}
