"""
.. sectionauthor:: Asher Bender <a.bender.dev@gmail.com>
.. codeauthor:: Asher Bender <a.bender.dev@gmail.com>
.. improved:: Gustavo Landfried <gustavolandfried@gmail.com>

.. |bool| replace:: :class:`.bool`
.. |callable| replace:: :func:`.callable`
.. |False| replace:: :data:`.False`
.. |float| replace:: :class:`.float`
.. |int| replace:: :class:`.int`
.. |ndarray| replace:: :class:`~numpy.ndarray`
.. |None| replace:: :data:`.None`
.. |True| replace:: :data:`.True`
.. |tuple| replace:: :func:`.tuple`

"""

# Copyright 2015 Asher Bender
# Copyright 2025 Asher Bender and Gustavo Landfried
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#The Bayesian linear model module was created using the following references:
# [1] Murphy, K. P., Machine learning: A probabilistic perspective, The MIT Press, 2012
# [2] Bishop, C. M, Pattern Recognition and Machine Learning (Information Science and Statistics), Jordan, M.; Kleinberg, J. & Scholkopf, B. (Eds.), Springer, 2006
# [3] Murphy, K. P., Conjugate Bayesian analysis of the Gaussian distribution, Department of Computer Science, The University of British Columbia, 2007

import numpy as np
import scipy.stats
from scipy.special import gammaln
import warnings
# https://docs.python.org/3/library/warnings.html#warnings.warn

# --------------------------------------------------------------------------- #
#                              Utility Functions
# --------------------------------------------------------------------------- #

def safe_logdet(A):
    """Compute log determinant in a numerically stable way.
    
    Args:
        A (ndarray): Square matrix
        
    Returns:
        float: Log determinant
    """
    sign, logdet = np.linalg.slogdet(A)
    
    if sign <= 0:
        if np.isclose(sign, 0):
            # Handle singular matrix case
            warnings.warn("Matrix is singular or near-singular in logdet calculation", RuntimeWarning)
            return -np.inf
        else:
            # This should not happen for positive definite matrices
            warnings.warn("Negative determinant encountered in logdet calculation", RuntimeWarning)
    
    return logdet


def safe_solve(A, b, rcond=1e-10):
    """Solve linear system Ax=b with robust handling of ill-conditioned matrices.
    
    Args:
        A (ndarray): Coefficient matrix
        b (ndarray): Right-hand side vector/matrix
        rcond (float): Cutoff for small singular values
        
    Returns:
        ndarray: Solution to Ax=b
    """
    # Check conditioning of the matrix
    cond = np.linalg.cond(A)
    
    if cond > 1e12:
        warnings.warn(f"Matrix is ill-conditioned (cond={cond:.2e}). Results may be inaccurate.", RuntimeWarning)
        
        # Use more stable pseudo-inverse for very ill-conditioned matrices
        x = np.linalg.lstsq(A, b, rcond=rcond)[0]
    else:
        try:
            # Try direct solve first
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fall back to least squares
            x = np.linalg.lstsq(A, b, rcond=rcond)[0]
            
    return x


# --------------------------------------------------------------------------- #
#                              Module Functions
# --------------------------------------------------------------------------- #
#
# The methods in this section implement the main equations and functionality of
# the Bayesian linear model. They are used as building blocks for the
# BayesianLinearModel() object - which is a convenience for error checking and
# maintaining variables (mostly the sufficient statistics).
#
# The functions in this section have been implemented as 'private'. That is,
# they are not visible in interactive sessions and are included in the sphinx
# documentation. However, since the names have not been fully mangled, they can
# still be accessed easily. They may be useful for other applications, however
# doing so comes with the caveat of less complete documentation and no error
# checking.


def _update(X, y, mu, S, alpha, beta):
    """Update sufficient statistics of the Normal-inverse-gamma distribution.

    Args:
      X (|ndarray|): (N x M) model inputs.
      y (|ndarray|): (N x 1) target outputs.
      mu (|ndarray|): Mean of the normal distribution.
      S (|ndarray|): Dispersion of the normal distribution (precision matrix).
      alpha (|float|): Shape parameter of the inverse Gamma distribution.
      beta (|float|): Scale parameter of the inverse Gamma distribution.

    Returns:
      |tuple|: The updated sufficient statistics are return as a tuple (mu, S,
           alpha, beta)

    """
    # Store prior parameters.
    mu_0 = mu
    S_0 = S

    # Update precision (Eq 7.71 ref [1], modified for precision).
    S = S_0 + np.dot(X.T, X)

    # Update mean (Eq 7.70 ref[1], modified for precision).
    #
    # Use more stable solver instead of direct Cholesky
    #
    # https://stackoverflow.com/questions/64831266/all-eigenvalues-are-positive-still-np-linalg-cholesky-is-giving-error-that-mat
    #
    # For matrices of moderate size, the performance difference is generally minimal. Cholesky is very stable for well-conditioned positive definite matrices, but fails completely if the matrix is not positive definite or close to singular.
    b = S_0.dot(mu_0) + X.T.dot(y)
    mu = safe_solve(S, b)

    # Update shape parameter (Eq 7.72 ref [1]).
    N = X.shape[0]
    alpha += N / 2.0

    # Update scale parameter (Eq 7.73 ref [1]).
    beta += 0.5 * (
        mu_0.T.dot(S_0.dot(mu_0)) +
        y.T.dot(y) -
        mu.T.dot(S.dot(mu))
    ).item()

    # Ensure beta is positive (numerical stability)
    beta = max(beta, 1e-10)

    return mu, S, alpha, beta


def _uninformative_fit(X, y, reg=1e-16):
    """Initialise sufficient statistics using an uninformative prior with regularization.

    Args:
      X (|ndarray|): (N x M) model inputs.
      y (|ndarray|): (N x 1) target outputs.
      reg (float): Regularization parameter for numerical stability.

    Returns:
      |tuple|: The updated sufficient statistics are return as a tuple (mu, S,
           alpha, beta)

    """

    N, D = X.shape

    # Add regularization for numerical stability
    XX = np.dot(X.T, X) + reg * np.eye(D)

    # Use stable solver
    mu = safe_solve(XX, np.dot(X.T, y))
    
    # Compute inverse with care
    try:
        V = np.linalg.inv(XX)
    except np.linalg.LinAlgError:
        # Fall back to pseudo-inverse for numerical stability
        warnings.warn("Using pseudo-inverse instead of XX calculation due to numerical issues", RuntimeWarning)
        # For a system Ax = b that has no exact solution, pinv provides the solution that minimises ‖Ax - b‖ with the lowest norm ‖x‖.
        # Unlike inv(), which throws error with singular matrices, pinv() always provides a 'best approximation' to the inverse.
        V = np.linalg.pinv(XX)

    # Uninformative prior parameters
    alpha = max(float(N - D) / 2.0, 1.0 + 1e-8) # Ensure alpha > 1 for numerical stability
    # An Inverse-Gamma distribution with shape parameter alpha, the mean exists only if alpha > 1 and the variance exists only if algha > 2. By ensuring that alpha = 1.0, we guarantee that the posterior distribution at least has a finite mean.

    # Careful calculation of residual sum of squares
    residuals = y - np.dot(X, mu)
    beta = 0.5 * np.sum(residuals**2)

    # Ensure beta is positive
    beta = max(beta, 1e-10)

    return mu, V, alpha, beta


def _predict_mean(X, mu):
    """Calculate posterior predictive mean.

    Args:
      X (|ndarray|): (N x M) input query locations to perform prediction.
      mu (|ndarray|): Mean of the normal distribution.

    Returns:
      |ndarray|: posterior mean

    """
    # Calculate mean.
    #     Eq 7.76 ref [1]
    return np.dot(X, mu)


def _predict_variance(X, S, alpha, beta):
    """Calculate posterior predictive variance.

    Args:
      X (|ndarray|): (N x M) input query locations to perform prediction.
      S (|ndarray|): Dispersion of the normal distribution (precision matrix).
      alpha (|float|): Shape parameter of the inverse Gamma distribution.
      beta (|float|): Scale parameter of the inverse Gamma distribution.

    Returns:
      |ndarray|: posterior standard deviation

    """
    # Note that the scaling parameter is not equal to the variance in the
    # general case. In the limit, as the number of degrees of freedom reaches
    # infinity, the scale parameter becomes equivalent to the variance of a
    # Gaussian.
    
    # Use stable solver instead of direct inversion

    uw = np.dot(X, np.linalg.solve(S, X.T))
    S_hat = (beta / alpha) * (np.eye(len(X)) + uw)
    S_hat = np.sqrt(np.diag(S_hat))

    return S_hat
    #XSinv = safe_solve(S, X.T).T
    #uw = np.dot(X, XSinv)
    # Ensure variance scaling is positive
    #scale_factor = max(beta / alpha, 1e-10)
    #S_hat = scale_factor * (np.eye(len(X)) + uw)
    #S_hat = np.sqrt(np.diag(S_hat))
    #return S_hat


def _posterior_likelihood(y, m_hat, S_hat, alpha, log=False):
    """Calculate posterior predictive data likelihood.

    Args:
      y (|ndarray|): (N x 1) output query locations.
      m_hat (|ndarray|): Predicted mean.
      S_hat (|ndarray|): Predicted variance.
      alpha (|float|): Shape parameter of the inverse Gamma distribution.
      log (|bool|, *optional*): Set to |True| to return the log-likelihood.

    Returns:
      |ndarray|: posterior likelihood or log-likelihood

    """
    # Ensure degrees of freedom is at least 1 for numerical stability
    df = max(2 * alpha, 1.0)
    
    # Ensure scale is positive
    S_hat = np.maximum(S_hat, 1e-10)

    if log:
        q = scipy.stats.t.logpdf(y, df=df, loc=m_hat, scale=S_hat)
    else:
        q = scipy.stats.t.pdf(y, df=df, loc=m_hat, scale=S_hat)

    return q


def _model_evidence(N, S_N, alpha_N, beta_N,
                   S_0=None, alpha_0=None, beta_0=None,
                   log=True, fallback=False):
    """Return log marginal likelihood of the data (model evidence).

    Note, if any of the optional parameters are set to |None|, their
    uninformative value will be used.

    Args:
      N (|int|): Number of observations.
      S_N (|ndarray|): Dispersion of the normal distribution (precision matrix).
      alpha_N (|float|): Shape parameter of the inverse Gamma distribution.
      beta_N (|float|): Scale parameter of the inverse Gamma distribution.
      S_0 (|ndarray|, *optional*): Prior dispersion of the normal distribution.
      alpha_0 (|float|, *optional*): Prior shape parameter of the inverse Gamma distribution.
      beta_0 (|float|, *optional*): Prior scale parameter of the inverse Gamma distribution.
      log (|bool|, *optional*): Set to |False| to return the (non-log) likelihood.
      fallback (|bool|, *optional*): Set to |True| to enable the fallback based on eigenvalues.

    Returns:
      |float|: The log marginal likelihood is returned.

    """
    # The likelihood can be broken into simpler components:
    # (Eq 3.118 ref [2], Eq 203 ref [3])
    #
    #     pdf = A * B * C * D
    #
    # where:
    #
    #     A = 1 / (2 * pi)^(N/2)
    #     B = (b_0 ^ a_0) / (b_N ^ a_N)
    #     C = gamma(a_N) / gamma(a_0)
    #     D = det(S_N)^(1/2) / det(S_0)^(1/2)
    #
    # Using log probabilities:
    #
    #     pdf = A + B + C + D
    #
    # where:
    #
    #     log(A) = -0.5 * N * ln(2 * pi)
    #     lob(B) = a_0 * ln(b_0) - a_N * ln(b_N)
    #     log(C) = gammaln(a_N) - gammaln(a_0)
    #     log(D) = ln(det(S_N)^0.5) - ln(det(S_0)^0.5)
    #

    A = -0.5 * N * np.log(2 * np.pi)

    # Check beta values are valid
    beta_N = max(beta_N, 1e-10)  # Ensure positive

    # Prior value specified.
    if beta_0 is not None:
        beta_0 = max(beta_0, 1e-10)  # Ensure positive
        B = alpha_0 * np.log(beta_0) - alpha_N * np.log(beta_N)

    # Approximate uninformative prior.
    else:
        B = -alpha_N * np.log(beta_N)

    # Prior value specified.
    if alpha_0 is not None:
        C = gammaln(alpha_N) - gammaln(alpha_0)

    # Approximate uninformative prior.
    else:
        C = gammaln(alpha_N)

    # Compute log determinants safely
    try:
        # Convert precision to covariance
        S_N_cov = np.linalg.inv(S_N) 
        logdet_SN = safe_logdet(S_N_cov)
        
        # Prior value specified
        if S_0 is not None:
            # Convert precision to covariance
            S_0_cov = np.linalg.inv(S_0)
            logdet_S0 = safe_logdet(S_0_cov)
            D = 0.5 * logdet_SN - 0.5 * logdet_S0
        # Approximate uninformative prior
        else:
            D = 0.5 * logdet_SN
    except Exception as e:
        warnings.warn(f"Error in logdet calculation: {str(e)}. Using fallback method.", RuntimeWarning)
        # Fallback method using eigenvalues
        if fallback:
            try:
                eigvals_N = np.linalg.eigvalsh(S_N)
                # Avoid negative eigenvalues
                eigvals_N = np.maximum(eigvals_N, 1e-10)
                logdet_SN = np.sum(np.log(eigvals_N))

                if S_0 is not None:
                    eigvals_0 = np.linalg.eigvalsh(S_0)
                    eigvals_0 = np.maximum(eigvals_0, 1e-10)
                    logdet_S0 = np.sum(np.log(eigvals_0))
                    D = 0.5 * logdet_SN - 0.5 * logdet_S0
                else:
                    D = 0.5 * logdet_SN
            except:
                # Last resort fallback
                warnings.warn("Critical failure in log determinant calculation. Returning approximate evidence.", RuntimeWarning)
                D = 0.0

    result = A + B + C + D
    
    # Check for invalid values
    if not np.isfinite(result):
        warnings.warn(f"Invalid model evidence value: {result}. Components: A={A}, B={B}, C={C}, D={D}", RuntimeWarning)
        if fallback:
            # Return a reasonable fallback value
            result = -1e10 if log else 0.0

    if log:
        return result
    else:
        return np.exp(result)


def _negative_log_marginal_likelihood(params, basis, X, y):
    """Return the negative log marginal likelihood.

    Note, this method is primarily for use within a convex optimiser
    (e.g. scipy.optimize.minimize).

    Args:
      basis (|callable|): Function for performing basis function expansion.
      params (|ndarray|): (N,) non-linear basis function parameters.
      X (|ndarray|): (N x M) model inputs.
      y (|ndarray|): (N x 1) target outputs.

    Returns:
      |float|: the negative log marginal likelihood.

    """
    try:
        phi = basis(X, params)
        mu, S, alpha, beta = _uninformative_fit(phi, y)

        m_hat = _predict_mean(phi, mu)
        S_hat = _predict_variance(phi, np.linalg.inv(S), alpha, beta)
        
        # Use log-likelihood for better numerical stability
        nll = -np.sum(_posterior_likelihood(y, m_hat, S_hat, alpha, log=True))
        
        # Check for invalid values
        if not np.isfinite(nll):
            # Return a large value to guide optimization away from this point
            return 1e10
            
        return nll
    except Exception as e:
        # Print diagnostic information
        print(f"Error in _negative_log_marginal_likelihood() calculation with params {params}: {str(e)}")
        # Return a large value
        return 1e10

# --------------------------------------------------------------------------- #
#                               Module Objects
# --------------------------------------------------------------------------- #


class BayesianLinearModel(object):
    r"""Bayesian linear model.

    Instantiate a Bayesian linear model. If no sufficient statistics are
    supplied at initialisation, the following uninformative semi-conjugate
    prior will be used:

    .. math::

          \mathbf{w}_0 &= \mathbf{0}                 \\
          \mathbf{V_0} &= (1/reg)\mathbf{I}   \\
          a_0          &= 1.0                        \\
          b_0          &= 1.0                        \\

    Where tau is a small regularization parameter. This differs from the original
    completely uninformative prior to provide better numerical stability.

    The sufficient statistics will be initialised during the first call to
    :py:meth:`.update` where the dimensionality of the problem can be inferred
    from the data.

    If the dimensionality of the problem ``D``, ``location`` or ``dispersion``
    are specified, initialisation will occur immediately. Uninformative values
    will be used for any unspecified parameters. Initialising the sufficient
    statistics immediately has the minor advantage of performing error checking
    before the first call to :py:meth:`.update`.

    Args:
      basis (|callable|): Function for performing basis function expansion on
        the input data.
      D (|int|, *optional*): Dimensionality of problem, after basis function
        expansion. If this value is supplied, it will be used for error
        checking when the sufficient statistics are initialised. If it is not
        supplied, the dimensionality of the problem will be inferred from
        either ``location``, ``dispersion`` or the first call to
        :py:meth:`.update`.
      location (|ndarray|, *optional*): Prior mean (:math:`\mathbf{w}_0`) of
        the normal distribution. Set to |None| to use uninformative value.
      dispersion (|ndarray|, *optional*): Prior dispersion
        (:math:`\mathbf{V}_0`) of the normal distribution. Set to |None| to use
        uninformative value.
      shape (|float|, *optional*): Prior shape parameter (:math:`a_0`) of the
        inverse Gamma distribution. Set to |None| to use uninformative value.
      scale (|float|, *optional*): Prior scale parameter (:math:`b_0`) of the
        inverse Gamma distribution. Set to |None| to use uninformative value.
      reg (|float|, *optional*): Regularization parameter to improve numerical 
        stability. Default is 1e-6.

    Raises:
      ~exceptions.Exception: If any of the input parameters are invalid.

    """

    def __init__(self, basis, D=None, location=None, dispersion=None, shape=None, scale=None, reg=1e-6):

        # Ensure the basis function expansion is a callable function.
        self.__basis = basis
        self.__basis_params = None
        if not callable(basis):
            msg = "The input 'basis' must be a callable function."
            raise Exception(msg)

        # Number of observations.
        self.__D = D
        self.__N = 0
        self.__reg = reg

        # Store prior.
        self.__mu_0 = location
        self.__S_0 = dispersion
        self.__alpha_0 = shape
        self.__beta_0 = scale

        # Work with precision if variance was provided.
        try:
            self.__S_0 = np.linalg.inv(self.__S_0)
        except:
            pass

        # Reset sufficient statistics.
        self.__mu_N = None
        self.__S_N = None
        self.__alpha_N = None
        self.__beta_N = None

        # Sufficient statistics have not been validated. Flag object as
        # uninitialised.
        self.__initialised = False

        # Attempt to initialise object from user input (either the
        # dimensionality 'D' or the sufficient statistics). If the object can
        # be initialise and there is an error, hault early. If the object
        # cannot be initialised, wait until a call to :py:meth:update (do not
        # throw error).
        try:
            self.__initialise(D=self.__D)
        except Exception as e:
            raise Exception(f"Error during initialization: {str(e)}")

    @property
    def size(self):
        if self.__N is None:
            warnings.warn("The model is not initialised. Returning None for shape", RuntimeWarning)
        return self.__N

    @property
    def location(self):
        return self.__mu_N

    @property
    def dispersion(self):
        """Return covariance matrix (inverse of precision matrix)"""
        try:
            return np.linalg.inv(self.__S_N)
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse for numerical stability
            warnings.warn("Using pseudo-inverse for covariance calculation due to numerical issues", RuntimeWarning)
            return np.linalg.pinv(self.__S_N)

    @property
    def shape(self):
        return self.__alpha_N

    @property
    def scale(self):
        return self.__beta_N

    def __initialise(self, D=None):
        """Initialise sufficient statistics of the distribution.

        This method initialises the sufficient statistics of the multivariate
        normal distribution if they have not been specified. If values have
        been specified, they are checked to ensure the dimensionality has been
        specified correctly.

        If values have not been specified, weakly informative values are used
        for better numerical stability (Section 7.6.3.2 of [1]):

            m = zero(D, 1)
            V = (1/reg) * eye(D)  # small regularization
            alpha = 1.0           # weakly informative
            beta = 1.0            # weakly informative

        """
        # Infer dimensionality...
        if D is None:
            # From the location parameter.
            if isinstance(self.__mu_0, np.ndarray) and self.__mu_0.ndim == 2:
                self.__D = self.__mu_0.shape[0]

            # From the dispersion parameters.
            elif isinstance(self.__S_0, np.ndarray) and self.__S_0.ndim == 2:
                self.__D = self.__S_0.shape[0]

            # Cannot infer the dimensionality, it has not been specified. Exit
            # initialisation.
            else:
                return None

        # Check the input dimensionality is a positive scalar.
        elif not np.isscalar(D) or D <= 0:
            msg = 'The input dimension must be a positive scalar.'
            raise Exception(msg)

        # Store dimensionality of the data.
        elif self.__D is None:
            self.__D = int(D)

        # If the location parameter has not been set, use zeros
        if self.__mu_0 is None:
            self.__mu_N = np.zeros((self.__D, 1))

        # Check that the location parameter is an array.
        elif (not isinstance(self.__mu_0, np.ndarray) or self.__mu_0.ndim != 2
              or self.__mu_0.shape[1] != 1):
            msg = 'The location parameter must be a (D x 1) numpy array.'
            raise Exception(msg)

        # Check the location parameter has the same dimensional as the input
        # data (after basis function expansion).
        elif self.__mu_0.shape[0] != self.__D:
            msg = 'The location parameter is a ({0[0]} x {0[1]}) matrix. '
            msg += 'The problem is {1}-dimensional. The location parameter '
            msg += 'must be a ({1} x 1) matrix.'
            raise Exception(msg.format(self.__mu_0.shape, self.__D))

        # User location is valid. Set location to specified value.
        else:
            self.__mu_N = self.__mu_0

        # If the dispersion parameter has not been set, use a weakly informative prior
        # with small regularization for numerical stability
        if self.__S_0 is None:
            self.__S_N = self.__reg * np.eye(self.__D)

        # Check that the dispersion parameter is an array.
        elif not isinstance(self.__S_0, np.ndarray) or self.__S_0.ndim != 2:
            msg = 'The dispersion parameter must be a (D x D) numpy array.'
            raise Exception(msg)

        # Check the dispersion parameter has the same dimensional as the input
        # data (after basis function expansion).
        elif ((self.__S_0.shape[0] != self.__D) and
              (self.__S_0.shape[1] != self.__D)):
            msg = 'The dispersion parameter is a ({0[0]} x {0[1]}) matrix. '
            msg += 'The design matrix (input data after basis function '
            msg += 'expansion) is {1}-dimensional. The dispersion parameter '
            msg += 'must be a ({1} x {1}) matrix.'
            raise Exception(msg.format(self.__S_0.shape, self.__D))

        # Convert covariance into precision.
        else:
            self.__S_N = self.__S_0

        # Use weakly informative shape for better numerical stability
        if self.__alpha_0 is None:
            self.__alpha_N = 1.0

        # Check the shape parameter is greater than zero.
        elif not np.isscalar(self.__alpha_0) or self.__alpha_0 <= 0:
            msg = 'The shape parameter must be greater than zero.'
            raise Exception(msg)

        # User shape is valid. Set shape to specified value.
        else:
            self.__alpha_N = self.__alpha_0

        # Use weakly informative scale for better numerical stability
        if self.__beta_0 is None:
            self.__beta_N = 1.0

        # Check the scale parameter is greater than zero.
        elif not np.isscalar(self.__beta_0) or self.__beta_0 <= 0:
            msg = 'The scale parameter must be greater than zero.'
            raise Exception(msg)

        # User scale is valid. Set scale to specified value.
        else:
            self.__beta_N = self.__beta_0

        # The sufficient statistics have been validated. Prevent object from
        # checking the sufficient statistics again.
        self.__initialised = True

    def reset(self):
        """Reset sufficient statistics to prior values."""

        # Force the model to reset the sufficient statistics.
        self.__initialised = False

        # Erase current sufficient statistics.
        self.__mu_N = None
        self.__S_N = None
        self.__alpha_N = None
        self.__beta_N = None

        # Attempt to initialise the sufficient statistics from the prior.
        try:
            self.__initialise()
        except Exception as e:
            raise Exception(f"Error during reset: {str(e)}")

    def __design_matrix(self, X):
        """Perform basis function expansion to create design matrix."""

        # Perform basis function expansion without parameters.
        if self.__basis_params is None:
            try:
                return self.__basis(X)
            except Exception as e:
                msg = 'Could not perform basis function expansion with the '
                msg += 'function %s\n\n' % str(self.__basis)
                msg += 'Error thrown:\n %s' % str(e)
                raise Exception(msg)

        # Perform basis function expansion WITH parameters.
        else:
            try:
                return self.__basis(X, self.__basis_params)
            except Exception as e:
                msg = 'Could not perform basis function expansion with '
                msg += 'parameters and the function %s\n\n' % str(self.__basis)
                msg += 'Error thrown:\n %s' % str(e)
                raise Exception(msg)

    def empirical_bayes(self, x0, X, y, method='L-BFGS-B', options=None):
        r"""Fit (non-linear) parameters to basis function using empirical Bayes.

        The optimal parameters are found by minimising the negative log
        marginal likelihood. A weakly informative prior is used for a robust optimization approach.

        Args:
          x0 (|ndarray|): (N,) starting (non-linear) basis function parameters.
          X (|ndarray|): (N x M) model inputs.
          y (|ndarray|): (N x 1) target outputs.
          method (str): Optimization method to use. Default is 'L-BFGS-B'.
          options (dict): Options to pass to the optimizer.

        Returns:
          |ndarray|: parameters of the basis function.

        """
        # Default optimization options for better convergence
        default_options = {'maxiter': 100, 'ftol': 1e-6}
        if options is not None:
            default_options.update(options)
        
        # Use robust optimization with bounds if using L-BFGS-B
        try:
            if method == 'L-BFGS-B':
                # Try to infer reasonable bounds if possible
                try:
                    bounds = [(-10, 10) for _ in range(len(x0))]
                    sol = scipy.optimize.minimize(_negative_log_marginal_likelihood,
                                              x0, args=(self.__basis, X, y),
                                              method=method, bounds=bounds,
                                              options=default_options)
                except:
                    # Fall back to unbounded optimization
                    sol = scipy.optimize.minimize(_negative_log_marginal_likelihood,
                                              x0, args=(self.__basis, X, y),
                                              method=method, options=default_options)
            else:
                # Use specified method
                sol = scipy.optimize.minimize(_negative_log_marginal_likelihood,
                                          x0, args=(self.__basis, X, y),
                                          method=method, options=default_options)
        except Exception as e:
            # If optimization fails, try with more robust method
            warnings.warn(f"Optimization failed with {method}: {str(e)}. Trying with COBYLA.", RuntimeWarning)
            sol = scipy.optimize.minimize(_negative_log_marginal_likelihood,
                                      x0, args=(self.__basis, X, y),
                                      method='COBYLA', options={'maxiter': 200})

        # Check for convergence
        if not sol.success:
            warnings.warn(f"Optimization may not have converged: {sol.message}", RuntimeWarning)

        # Store optimal parameters.
        self.__basis_params = sol.x

        # Recover sufficient statistics from optimal parameters.
        phi = self.__design_matrix(X)
        self.__mu_N, self.__S_N, self.__alpha_N, self.__beta_N = \
            _uninformative_fit(phi, y, reg=self.__reg)

        # The sufficient statistics have been initialised. Prevent object from
        # checking the sufficient statistics again.
        self.__initialised = True

        return self.__basis_params

    def update(self, X, y):
        r"""Update sufficient statistics of the Normal-inverse-gamma distribution.

        .. math::

            \mathbf{w}_N &= \mathbf{V_N}\left(
                                \mathbf{V_0}^{-1}\mathbf{w}_0 +
                                \mathbf{\Phi}^T\mathbf{y}
                            \right)                                          \\
            \mathbf{V_N} &= \left(\mathbf{V_0}^{-1} +
                                  \mathbf{\Phi}^T\mathbf{\Phi}\right)^{-1}   \\
            a_N          &= a_0 + \frac{n}{2}                                \\
            b_N          &= b_0 + \frac{k}{2}
                            \left(\mathbf{w}_0^T\mathbf{V}_0^{-1}\mathbf{w}_0 +
                                  \mathbf{y}^T\mathbf{y} -
                                  \mathbf{w}_N^T\mathbf{V}_N^{-1}\mathbf{w}_N
                            \right)

        Args:
          X (|ndarray|): (N x M) model inputs.
          y (|ndarray|): (N x 1) target outputs.

        Raises:
          ~exceptions.Exception: If there are not enough inputs or the
            dimensionality of the data is wrong.

        """
        # Ensure inputs are valid objects and the same length.
        if (not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray) or
            (X.ndim != 2) or (y.ndim != 2) or (len(X) != len(y))):
            msg = 'X must be a (N x M) matrix and y must be a (N x 1) vector.'
            raise Exception(msg)

        # Check for NaN or infinite values
        if np.any(np.isnan(X)) or np.any(np.isnan(y)) or np.any(np.isinf(X)) or np.any(np.isinf(y)):
            msg = 'Input data contains NaN or infinite values.'
            raise Exception(msg)

        # Perform basis function expansion.
        phi = self.__design_matrix(X)

        # Get size of input data.
        N, D = phi.shape
        self.__N += N

        # Check sufficient statistics are valid (only once).
        if not self.__initialised:
            self.__initialise(D)

        # Check dimensions of input data.
        if self.__D != D:
            msg = 'The input data, after basis function expansion, is '
            msg += '{0}-dimensional. Expected {1}-dimensional data.'
            raise Exception(msg.format(D, self.__D))

        # Update sufficient statistics.
        self.__mu_N, self.__S_N, self.__alpha_N, self.__beta_N = \
            _update(phi, y, self.__mu_N, self.__S_N,
                   self.__alpha_N, self.__beta_N)

    def predict(self, X, y=None, variance=False, ci_level=0.95):
        r"""Calculate posterior predictive values.

        Given a new set of test inputs, :math:`\tilde{\mathbf{X}}`, predict the
        output value. The predictions are T-distributed according to:

        .. math::

            p\left(\tilde{\mathbf{y}} \vert
                          \tilde{\mathbf{X}}, \mathcal{D} \right) =
            \mathcal{T}\left(\tilde{\mathbf{y}} \; \big\vert \;
                             \tilde{\mathbf{\Phi}}\mathbf{w}_N,
                             \frac{b_N}{a_N}
                             \left(\mathbf{I} +
                                   \tilde{\mathbf{\Phi}}\mathbf{V}_N\tilde{\mathbf{\Phi}}^T
                             \right),
                             2a_N
                       \right)

        The data likelihood can also be requested by specifying both a set of
        test inputs, :math:`\tilde{\mathbf{X}}` and a set of test output,
        :math:`\tilde{\mathbf{y}}`, locations.

        Args:
          X (|ndarray|): (N x M) input query locations
            (:math:`\tilde{\mathbf{X}}`) to perform prediction.
          y (|ndarray|, *optional*): (K x 1) output query locations
            (:math:`\tilde{\mathbf{y}}`) to request data likelihood.
          variance (|bool|, *optional*): set to |True| to return the confidence
            intervals. Default is set to |False|.
          ci_level (|float|, *optional*): Confidence interval level (e.g., 0.95 for 95% CI).
            Default is 0.95.

        Returns:
          |ndarray| or |tuple|:
            * |ndarray|: By default only the predicted means are returned as a
              (N x 1) array.
            * (|ndarray|, |ndarray|): If ``variance`` is set to |True| a tuple
              is returned containing both the predicted means (N x 1) and the
              confidence intervals (N x 1).
            * (|ndarray|, |ndarray|, |ndarray|): If ``y`` is set, the value of
              ``variance`` is ignored and the predicted means and confidence
              intervals are returned. The final returned value is
              either a (N x 1) array or (K x N) matrix of likelihood values. If
              ``X`` and ``y`` are the same length (N = K), the result is
              returned as an array. If ``X`` and ``y`` are NOT the same length
              (N != K), a matrix is returned where each row represents an
              element in ``y`` and each column represents a row in ``X``.

        Raises:
          ~exceptions.Exception: If the sufficient statistics have not been
            initialised with observed data. Call :py:meth:`.update` first.

        """
        # Ensure the sufficient statistics have been initialised.
        if not self.__initialised:
            msg = 'The sufficient statistics need to be initialised before '
            msg += "calling 'predict()'. Run 'update()' first."
            raise Exception(msg)

        # Check confidence interval level is valid
        if not (0 < ci_level < 1):
            msg = 'Confidence interval level must be between 0 and 1.'
            raise Exception(msg)

        # Perform basis function expansion.
        phi = self.__design_matrix(X)

        # Calculate mean.
        m_hat = _predict_mean(phi, self.__mu_N)

        # Calculate variance.
        if (y is not None) or variance:
            S_hat = _predict_variance(phi,
                                     self.__S_N,
                                     self.__alpha_N,
                                     self.__beta_N)

            # Calculate a one sided confidence interval based on the t-distribution
            # For a tabulation of values see:
            #     http://en.wikipedia.org/wiki/Student%27s_t-distribution#Confidence_intervals
            #
            # Note: If the number of degrees of freedom is equal to one, the
            #       distribution is equivalent to the Cauchy distribution. As
            #       the number of degrees of freedom approaches infinite, the
            #       distribution approaches a Gaussian distribution.
            #
            alpha = 1.0 - ci_level
            ci = scipy.stats.t.ppf(1.0 - alpha/2, 2 * self.__alpha_N)

            # Return mean and confidence interval.
            if y is None:
                return (m_hat, ci * S_hat[:, np.newaxis])

            # Return mean, confidence interval and likelihood.
            else:
                N = phi.shape[0]
                K = y.size

                # Return array.
                if N == K:
                    q = _posterior_likelihood(y.squeeze(),
                                            m_hat.squeeze(),
                                            S_hat.squeeze(),
                                            self.__alpha_N)

                # Return matrix result.
                else:
                    q = _posterior_likelihood(y.reshape((K, 1)),
                                            m_hat.reshape((1, N)),
                                            S_hat.reshape((1, N)),
                                            self.__alpha_N)

                return (m_hat, ci * S_hat[:, np.newaxis], q)

        else:
            return m_hat

    def evidence(self, log=True, robust=True):
        r"""Return log marginal likelihood of the data (model evidence).

        The log marginal likelihood is calculated by taking the log of the
        following equation:

        .. math::

            \renewcommand{\det} [1]{{\begin{vmatrix}#1\end{vmatrix}}}

            p\left(\mathcal{D} \right) = \frac{1}{2\pi^\frac{N}{2}}
                                         \frac{\det{V_N}^\frac{1}{2}}
                                              {\det{V_0}^\frac{1}{2}}
                                         \frac{b_0^{a_0}}
                                              {b_N^{a_N}}
                                         \frac{\Gamma\left(a_N\right)}
                                              {\Gamma\left(a_0\right)}

        Args:
          log (|bool|, *optional*): Set to |False| to return the (non-log)
              marginal likelihood.
          robust (|bool|, *optional*): Set to |False| to use the original
              calculation method (less numerically stable).

        Returns:
          |float|: The (log) marginal likelihood is returned.

        Raises:
          ~exceptions.Exception: If the sufficient statistics have not been
            initialised with observed data. Call :py:meth:`.update` first.

        """
        # Ensure the sufficient statistics have been initialised.
        if not self.__initialised:
            msg = 'The sufficient statistics need to be initialised before '
            msg += "calling 'evidence()'. Run 'update()' first."
            raise Exception(msg)

        if robust:
            # Use more robust calculation method
            try:
                # Define components of the evidence calculation based on our sufficient statistics
                N_obs = self.__N
                
                # Convert precision to covariance matrices
                try:
                    S_N_cov = np.linalg.inv(self.__S_N)
                    if self.__S_0 is not None:
                        S_0_cov = np.linalg.inv(self.__S_0)
                except np.linalg.LinAlgError:
                    # Fall back to pseudo-inverse for numerical stability
                    warnings.warn("Using pseudo-inverse for covariance calculation due to numerical issues", RuntimeWarning)
                    S_N_cov = np.linalg.pinv(self.__S_N)
                    if self.__S_0 is not None:
                        S_0_cov = np.linalg.pinv(self.__S_0)
                
                # Calculate components
                A = -0.5 * N_obs * np.log(2 * np.pi)
                
                # Use slogdet for numerical stability
                sign_N, logdet_N = np.linalg.slogdet(S_N_cov)
                
                if sign_N <= 0:
                    warnings.warn("Covariance matrix has non-positive determinant. Evidence calculation may be inaccurate.", RuntimeWarning)
                    # Handle the case where determinant is negative or zero
                    logdet_N = -np.inf if sign_N == 0 else np.log(np.abs(np.linalg.det(S_N_cov)))
                
                # Calculate determinant component
                if self.__S_0 is not None:
                    sign_0, logdet_0 = np.linalg.slogdet(S_0_cov)
                    if sign_0 <= 0:
                        warnings.warn("Prior covariance matrix has non-positive determinant.", 
                                     RuntimeWarning)
                        logdet_0 = -np.inf if sign_0 == 0 else np.log(np.abs(np.linalg.det(S_0_cov)))
                    D = 0.5 * (logdet_N - logdet_0)
                else:
                    D = 0.5 * logdet_N
                
                # Calculate shape-scale components
                if self.__beta_0 is not None:
                    B = self.__alpha_0 * np.log(self.__beta_0) - self.__alpha_N * np.log(self.__beta_N)
                else:
                    B = -self.__alpha_N * np.log(self.__beta_N)
                
                if self.__alpha_0 is not None:
                    C = gammaln(self.__alpha_N) - gammaln(self.__alpha_0)
                else:
                    C = gammaln(self.__alpha_N)
                
                # Combine for final evidence
                evidence_val = A + B + C + D
                
                # Check for validity
                if not np.isfinite(evidence_val):
                    warnings.warn(f"Invalid evidence value: {evidence_val}. Components: A={A}, B={B}, C={C}, D={D}", 
                                 RuntimeWarning)
                    evidence_val = -np.inf if log else 0.0
                
                return evidence_val if log else np.exp(evidence_val)
                
            except Exception as e:
                warnings.warn(f"Error in robust evidence calculation: {str(e)}. Falling back to standard method.", 
                             RuntimeWarning)
                # Fall back to standard method
                pass
        
        # Use standard method
        return _model_evidence(self.__N,
                              self.__S_N, self.__alpha_N, self.__beta_N,
                              self.__S_0, self.__alpha_0, self.__beta_0,
                              log=log)

    def random(self, samples=1):
        r"""Draw a random model from the posterior distribution.

        The model parameters are T-distributed according to the following
        posterior marginal:

        .. math::

            p\left(\mathbf{w} \vert \mathcal{D} \right) =
            \mathcal{T}\left(
                           \mathbf{w}_N, \frac{b_N}{a_N}\mathbf{V}_N, 2a_N
                       \right)

        Args:
          samples (|int|, *optional*): number of random samples to return.

        Returns:
          |ndarray|: Return (NxD) random samples from the model weights
            posterior. Each row is a D-dimensional vector of random model
            weights.

        Raises:
          ~exceptions.Exception: If the sufficient statistics have not been
            initialised with observed data. Call :py:meth:`.update` first.

        """
        # Ensure the sufficient statistics have been initialised.
        if not self.__initialised:
            msg = 'The sufficient statistics need to be initialised before '
            msg += "calling 'random()'. Run 'update()' first."
            raise Exception(msg)

        # The posterior over the model weights is a Student-T distribution. To
        # generate random models, sample from the posterior marginals.
        #
        #     Eq 7.75 ref [1]

        # Try to use the standard approach first
        try:
            # Draw random samples from the inverse gamma distribution (1x1xN).
            r = scipy.stats.invgamma.rvs(self.__alpha_N,
                                        scale=self.__beta_N,
                                        size=samples).reshape((1, 1, samples))

            # Create multiple multivariate scale matrices from random gamma samples
            # (DxDxN).
            try:
                sigma = np.linalg.inv(self.__S_N)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse for stability
                warnings.warn("Using pseudo-inverse for covariance calculation due to numerical issues", RuntimeWarning)
                sigma = np.linalg.pinv(self.__S_N)
                
            sigma = r * np.repeat(sigma[:, :, np.newaxis], samples, axis=2)

            # Draw random samples from the standard univariate normal distribution
            # (1xDxN).
            rn = np.random.normal(size=(1, self.__D, samples))

            # Create N random samples (1xD) drawn from multiple, random and unique
            # multivariate normal distributions.
            try:
                L = np.rollaxis(np.linalg.cholesky(sigma.T).T, 0, 2)
                sigma = np.dot(np.rollaxis(rn, 0, 2).T, L.T)
            except np.linalg.LinAlgError:
                # Fall back to eigendecomposition for stability
                warnings.warn("Using eigendecomposition for sampling due to Cholesky failure", 
                             RuntimeWarning)
                samples_list = []
                for i in range(samples):
                    # Extract the i-th covariance matrix
                    cov_i = sigma[:, :, i]
                    # Compute eigendecomposition
                    eigvals, eigvecs = np.linalg.eigh(cov_i)
                    # Ensure eigenvalues are positive
                    eigvals = np.maximum(eigvals, 1e-10)
                    # Transform standard normal samples
                    z = np.random.normal(size=self.__D)
                    x = eigvecs.dot(np.sqrt(eigvals) * z)
                    samples_list.append(x)
                    
                sigma = np.vstack(samples_list)
                return self.__mu_N.T + sigma
            
            # Return (NxD) samples drawn from multivariate-normal, inverse-gamma
            # distribution.
            return self.__mu_N.T + sigma
            
        except Exception as e:
            # Fall back to simpler sampling method if the vectorized approach fails
            warnings.warn(f"Using alternative sampling method due to error: {str(e)}", 
                         RuntimeWarning)
            
            # Alternative sampling approach
            samples_list = []
            
            for _ in range(samples):
                # Sample precision from inverse gamma
                prec = scipy.stats.invgamma.rvs(self.__alpha_N, scale=self.__beta_N)
                
                # Get covariance matrix
                try:
                    cov = np.linalg.inv(self.__S_N) * prec
                except np.linalg.LinAlgError:
                    warnings.warn("Using pseudo-inverse for covariance calculation due to numerical issues", RuntimeWarning)
                    cov = np.linalg.pinv(self.__S_N) * prec
                
                # Add a small jitter to ensure positive definiteness
                cov = cov + np.eye(self.__D) * 1e-8
                
                # Sample from multivariate normal
                try:
                    sample = np.random.multivariate_normal(
                        self.__mu_N.flatten(), cov
                    )
                except np.linalg.LinAlgError:
                    # Even more basic approach if that fails
                    z = np.random.normal(size=self.__D)
                    sample = self.__mu_N.flatten() + z * np.sqrt(np.diag(cov))
                    
                samples_list.append(sample)
                
            return np.vstack(samples_list)

    def get_diagnostics(self):
        """Return diagnostic information about the model.
        
        Returns:
            dict: Dictionary containing diagnostic information
        """
        if not self.__initialised:
            return {'initialised': False}
            
        try:
            # Calculate eigenvalues of precision matrix
            try:
                eigvals_S = np.linalg.eigvalsh(self.__S_N)
                cond_S = np.max(eigvals_S) / np.min(eigvals_S) if np.min(eigvals_S) > 0 else np.inf
            except:
                eigvals_S = None
                cond_S = np.inf
                
            # Get covariance matrix
            try:
                cov = np.linalg.inv(self.__S_N)
                eigvals_cov = np.linalg.eigvalsh(cov)
                cond_cov = np.max(eigvals_cov) / np.min(eigvals_cov) if np.min(eigvals_cov) > 0 else np.inf
            except:
                cov = None
                eigvals_cov = None
                cond_cov = np.inf
                
            return {
                'initialised': True,
                'observations': self.__N,
                'dimensions': self.__D,
                'shape_parameter': self.__alpha_N,
                'scale_parameter': self.__beta_N,
                'condition_number_precision': cond_S,
                'condition_number_covariance': cond_cov,
                'min_eigenvalue_precision': np.min(eigvals_S) if eigvals_S is not None else None,
                'min_eigenvalue_covariance': np.min(eigvals_cov) if eigvals_cov is not None else None,
            }
        except Exception as e:
            return {
                'initialised': True,
                'error': str(e),
                'observations': self.__N,
                'dimensions': self.__D,
            }
