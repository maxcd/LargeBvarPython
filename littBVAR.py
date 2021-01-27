from olsvar import *
import numpy as np
from numpy import linalg as la
from scipy.linalg.lapack import dpotrf as chol
from scipy.stats import invgamma, gamma, norm, invwishart

class BVARinfo:

    '''
    Class to store information for estimating the Minnesota prior
    BVARs in Banbura et.al (2010)

    '''

    def __init__(self, data=None, train_data=None, varlist=None, lags=None, ic=None, RW=None, shrink=None,
                 eval_names=None, ar_std=None, vareval=None, shock_pos=None, k=None,
                 impulse_pos=None, means=None, codes=None, title=None, params=None,
                 sigma=None, rel_fit=None, Xd=None, Yd=None, fit=None,
                 resid_covar_ols=None, train_means=None, train_ar_std=None,
                 impulse_std=None, impulse_std_ols=None, irf=None, upper=None,
                 lower=None):

        self.title = title               # name of thee model for printing
        self.data = data                 # data to estimate the model on
        self.varlist = varlist           # position in excel file of variables to include
        self.n = len(varlist)            # number of variables
        self.k = k                       # number of regressors per equation n*lags+1
        self.lags = lags                 # lag order
        self.ic = ic                     # information criteria for OLS model

        self.RW = RW                     # indication random walk prior or not
        self.shrink = shrink             # overall degree of shrinnkage "lambda"
        self.ar_std = ar_std             # resid variance of univariate ar(p) model for all variables
        self.means = means                # means of variables trough out estimaion sample
        self.Xd = Xd                     # dummies for regressors
        self.Yd = Yd                     # dummies for endogenous variables

        self.vareval = vareval          # boolean selection array to select variables for forecast comparison
        self.eval_names = eval_names    # names of variables to evaluate IRFs, forecasts and fit
        self.codes = codes              # mnemonics of variables

        self.train_data = train_data        # trainig data set for shrinkage determination
        self.train_ar_std = train_ar_std    # corresponding prior arguments for training sample
        self.train_means = train_means
        self.fit = fit
        self.resid_covar_ols = resid_covar_ols # fit in the training sample for optimal shrinkage

        self.impulse_pos = impulse_pos          # position of the shick of interst
        self.impulse_std = impulse_std          # std deviation for normalization to unit impact
        self.impulse_std_ols = impulse_std_ols  # same for the ols irf
        # self.shock_pos = shock_pos      # index value of the shock of interest
        self.irf = irf                  # mean impulse response
        self.upper = upper              # upper CB
        self.lower = lower              # lower CB

        self.params = params            # estimated slope coefficients and constant
        self.sigma = sigma              # covariance matrix of estimated residuals
        self.rel_fit = rel_fit          # relative fit for varevals agains random walk

def Litt_dummies(n, p, prior_args, det_first=True, sum_of_coefs=False):
    '''
    set up dummy variables for Minnesota prior for a VAR with intercept
    intercept should be ordered last in each equation

    IMPUTS
    ---
    n : int
        number of endogenous variables in the VAR

    p : int
        lag order

    prior_args : list or tuple

        contains:

            lam : int
                hyperparameter specifying the tightness of the prior

            omega : (n,) 1d_array
                list of zeros and ones indicating the prior mean for each
                variable in the variable. Should take 1 for persistent
                variables and 0 else.

            sigma : (n,) 1d_array
                list containing the residual standard deviations of individually fitted
                ar processes to each of the n endogenous variables.

            mu : (n,) 1d_array
                of sample averages of the individual variables (only needed for sum
                of coefficients prior)

    det_first : boolean
        if 'True' order the dummies for the constant in in front

    sum_of_coefs : boolean
        if 'True' sets up the dummis for the sum of coefficients prior too.

    OUTPUT
    ---
    Yd : nd_array
        if sum_of_coefs is 'True' it Yd has shape (np+2n+1) x (n)
        if sum_of_coefs is 'False' Yd has shape (np+n+1) x (n)

    Xd : np_array
        if sum_of_coefs is 'True' Xd has shape (np+2n+1) x (np+1)
        if sum_of_coefs is 'False' Xd has shape (np+n+1) x (np+1)
    '''

    if sum_of_coefs:
        lam, omega, sigma, mu = prior_args
        tau = 10 * lam
    else:
        lam, omega, sigma, mu = prior_args

    NP = n*p
    eps = 1e-5
    olam = omega / lam

    # diagonal for the litterman prior
    if sum_of_coefs:
        ydiag = sigma * olam
        y_sum_diag = omega * mu / tau

        Yd = np.concatenate([np.diag(ydiag),
                             np.zeros((n*(p-1),n)), # dummies for var coefficicnts
                             np.diag(y_sum_diag),   # adding the sum of coefficients block
                             np.diag(sigma)], axis=0) # for the covrariance matrix

        x_sum = np.kron(np.ones((1, p)), np.diag(y_sum_diag))
        Xd = np.zeros((NP+2*n, NP+1)) # larger dmension
        npn = NP+n
        Xd[NP:npn,:NP] = x_sum
        np.fill_diagonal(Xd[:NP,:NP], (np.outer(np.arange(1, p+1), sigma).ravel() / lam))
        Xd[-1, -1] = eps
    elif not sum_of_coefs:
        ydiag = sigma * olam
        Yd = np.concatenate([np.diag(ydiag), # for slope coefficients
                             np.zeros((n*(p-1),n)),
                             np.diag(sigma)], axis=0) # for the covariance

        Xd = np.zeros((NP+n, NP+1))
        np.fill_diagonal(Xd[:NP,:NP], (np.outer(np.arange(1, p+1), sigma).ravel() / lam))
        #Xd[-1, -1] = eps

    if det_first: # paste dummis for the constant in front
        Yd = np.concatenate([np.zeros((1, n)), Yd], axis=0)
        Xd = np.concatenate([np.zeros((1, NP+1)), Xd], axis=0)
        Xd[0, -1] = eps
    elif not det_first: # or in the back
        Yd = np.concatenate([Yd, np.zeros((1, n))], axis=0)
        Xd = np.concatenate([Xd, np.zeros((1, NP+1))], axis=0)
        Xd[-1, -1] = eps

    return Yd, Xd

def BVAR_litt(Y, X, lags, prior_args, det_first=True, sum_of_coefs=False):

    T, n = Y.shape
    k = X.shape[1]
    Yd, Xd = Litt_dummies(n, lags, prior_args, det_first=det_first,
                          sum_of_coefs=sum_of_coefs)

    # compute posterior mean for parameter estimates
    XtX_star = X.T @ X + Xd.T @ Xd
    XtY_star = X.T @ Y + Xd.T @ Yd

    beta = la.solve(XtX_star, XtY_star)
    e = Y - X @ beta
    ed = Yd - Xd @ beta
    mean_e = np.concatenate([ e , ed ], axis=0)
    Sig = (mean_e.T @ mean_e) / (T - k)

    # comute the fit
    SS = np.dot(e.T, e) / (T - k + 2)
    fit = np.diag(SS) / np.diag(np.cov(diff(Y).T))

    return beta, Sig, e

def BVAR_litt2(data, lags, prior_args, det_first=True, sum_of_coefs=False,
               ndraws=0, irh=0, sig_levels=None):

    Y, X = make_var_regressors(data, lags, det_first=det_first)

    T, n = Y.shape
    k = X.shape[1]
    Yd, Xd = Litt_dummies(n, lags, prior_args, det_first=det_first,
                          sum_of_coefs=sum_of_coefs)
    Td = Yd.shape[0]
    # compute posterior mean for parameter estimates
    XtX_star = X.T @ X + Xd.T @ Xd
    XtY_star = X.T @ Y + Xd.T @ Yd
    beta = la.solve(XtX_star, XtY_star)
    e = Y - X @ beta
    ed = Yd - Xd @ beta
    mean_e = np.concatenate([ e , ed ], axis=0)
    Sig = (mean_e.T @ mean_e) / (T+Td)

    # comute the fit
    SS = e.T @ e / (data.shape[0] - lags + 1)
    fit = 1 - np.diag(SS) / np.diag(np.cov(diff(data).T))

    return beta, Sig, fit

def BVAR_irf(data, lags, prior_args, det_first=True, sum_of_coefs=False,
               ndraws=200, irh=48, upper=[95], lower=[5], shocks=None):

    Y, X = make_var_regressors(data, lags, det_first=det_first)

    T, n = Y.shape
    k = X.shape[1]
    Yd, Xd = Litt_dummies(n, lags, prior_args, det_first=det_first,
                          sum_of_coefs=sum_of_coefs)
    Td = Yd.shape[0]
    # compute posterior mean for parameter estimates
    XtX_star = X.T @ X + Xd.T @ Xd
    XtY_star = X.T @ Y + Xd.T @ Yd
    beta = la.solve(XtX_star, XtY_star)
    e = Y - X @ beta
    ed = Yd - Xd @ beta
    mean_e = np.concatenate([ e , ed ], axis=0)
    SSR = (mean_e.T @ mean_e)
    Sig = SSR / (T+Td)
    mean_scale = np.diag(chol(Sig)[0])

    # compute thee prior arguments
    S0 = ed.T @ ed

    mean_irf = get_irfs(beta[:-1,:].T, n, lags, irh, Sig) / mean_scale[None,:,None]

    # compute credible sets
    CSiginv = chol(la.inv(SSR))[0]  # upper cholesky factor
    X_st = np.concatenate([Xd, X], axis=0)
    CXXinv = chol(la.inv(XtX_star))[0]

    beta_i = la.solve(X.T@X, X.T@Y)
    MCsample = np.zeros((ndraws, irh, n, n))
    for isim in range(ndraws):

        # draw residual covariance from invers wishart
        Z = norm.rvs(size=(T+Td+2-k, n))
        Zi = Z @ CSiginv
        Sig_i = la.inv(Zi.T @ Zi)
        CSig_i = chol(Sig_i)[0]

        # draw beta from multivariate normal
        temp = norm.rvs(size=beta.shape)
        beta_i = beta + CXXinv.T @ temp @ CSig_i
        irf_i = get_irfs(beta_i[:-1,:].T, n, lags, irh, Sig_i)
        imp_scale = np.diag(CSig_i)
        MCsample[isim,:,:,:] = irf_i / imp_scale[None,:,None]

    nupper = len(upper)
    ub = np.zeros((nupper, irh, n, n))
    nlower = len(lower)
    lb = np.zeros((nlower, irh, n, n))
    for ii in range(nupper):
        ub[ii,:,:,:] = np.percentile(MCsample, upper[ii], axis=0)
        lb[ii,:,:,:] = np.percentile(MCsample, lower[ii], axis=0)

    return mean_irf, ub, lb


def SURform(data, lags, det='c'):

    '''
    generate the regressors for a variate

    INPUT
    ---
    data : (T, n) nd-array

    lags : int
        desired number of lags as make_ar_regressors

    det : str
        specify deterministic regressors
        'c'(default) : include a constant as first regressor


    OUTPUT
    ---
    Yvec : (T*n, 1) nd-array
        Vecorized left hand side variables (current variables)

    Y : (T, n) nd-array
        not vectorized LHS variables

    Z : (T*n, n*k) nd-array
        regressor matrix in form as in Chan's notes, k=n*lags + 1
        being th number of regressors per equation in the case
        with only a constant.

    X : (T, k) nd-array
        most compact form of regressor matrix as in
        Helmut Luetkepohl's book and slides.
        The ith row contains the k regressors of period i starting
        with the constant.

    '''

    S, n = data.shape
    T = S - lags
    k = n * lags + 1 # number of params per equation

    X = data[lags-1:-1,:].copy()

    if lags >= 1:
        for i in range(2, lags+1):
            X = np.concatenate([X, data[lags-i:-i,:]], axis=1)

    if det == 'c':
        X = np.concatenate([np.ones([T,1]) ,X], axis=1)

    Z = np.eye(n)[:,:,None] * X[0,:].repeat(n).reshape(k, n).T
    for i in range(1, T):
        Z_next = np.eye(n)[:,:,None] * X[i,:].repeat(n).reshape(k, n).T
        Z = np.concatenate([Z, Z_next], axis=0)

    Y = data[lags:,]
    Yvec = np.hstack(Y).reshape(T*n,1)

    return Yvec, Y, Z.reshape((T*n, n*k)), X

def opt_shrinkage():

    Grid = np.concatenate([np.arange(0,5.01,0.025), np.ones(1)*50])


def sample_beta(X, y, Sig, prior_args):
    # sample beta from normal inverse wishart VAR in a Gibbs sampler
    nu0, S0, beta0, iVbeta, n, T, k = prior_args

    iSig = la.solve(Sig, np.eye(n))
    Xisig = X.T @ np.kron(np.eye(T), iSig)
    Kbeta = iVbeta + Xisig @ X
    beta_hat = la.solve(Kbeta, iVbeta @ beta0  + Xisig @ y)
    beta = beta_hat + la.solve(la.cholesky(Kbeta).T, norm.rvs(size=(n*k,1)))
    return beta

def sample_Sig(X, y, beta, prior_args):
    # sample residual covariance matrix from normal inverse wishart VAR in a Gibbs sampler
    nu0, S0, beta0, iVbeta, n, T, k = prior_args

    e = (y - X @ beta).reshape(n, T, order='F')
    Sig_hat = e @ e.T
    Sig = invwishart(df=nu0+T, scale=S0+Sig_hat).rvs()
    return Sig

def BVAR_Gibbs(data, lags, nsim, burnin, prior_args, print_step):
    S, n = data.shape
    T = S -lags
    k = n*lags + 1 # VAR with constant
    y, Y, X, Xtilde = SURform(data, lags)

    # storage matrices
    betamat = np.zeros((nsim, n*k))
    sigmat = np.zeros((nsim, n, n))

    # update arguments
    nu0, S0, beta0, iVbeta  = prior_args
    gibbs_args = nu0, S0, beta0, iVbeta, n, T, k

    # initialize the chain
    beta = la.solve(Xtilde.T @ Xtilde, Xtilde.T @ Y)
    e = Y - Xtilde @ beta
    Sig = e.T @ e / T
    iSig = la.solve(Sig, np.eye(n))
    beta = beta.reshape((n*k, 1), order='F') # column stack

    for isim in range(nsim+burnin):

        beta = sample_beta(X, y, Sig, gibbs_args)
        Sig = sample_Sig(X, y, beta, gibbs_args)

    # set values for the next iteration
        if isim >= burnin:
            row = isim - burnin
            betamat[row, :] = beta.flatten()
            sigmat[row,:,:] = Sig

        #print(count)
        if isim % print_step == 0:
            print('iteration no.', str(isim))

    print('terminated ', nsim+burnin, ' loops')
    return betamat, sigmat
