from numpy.linalg import det
from numpy.linalg import solve
import numpy as np
from numpy.linalg import inv, pinv
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats

def ldiff(x):
    '''
        take log difference along the zero axis
        i.e. different rows must correspond to different points in time
    '''
    logdiff = np.log(x[1:]) - np.log(x[:-1])
    return logdiff

def diff(x):
    diff = x[1:] - x[:-1]
    return diff

def covar(X0):
    T, n = X0.shape
    X = X0 - X0.mean(axis=0)
    fact = float(T - 1)
    return np.dot(X.T, X) / fact

def AIC(SigU, m, K, T):
    aic = np.log(det(SigU)) + 2/T * m * K ** 2
    return aic

def HQ(SigU, m, K, T):
    hq = np.log(det(SigU)) + 2 * np.log(np.log(T))/T * m * K ** 2
    return hq

def SC(SigU, m, K, T):
    sc = np.log(det(SigU)) + np.log(T)/T * m * K ** 2
    return sc

def make_reg_same_sample(Y, maxlag, m):
    nobs, K = Y.shape
    T = nobs - maxlag
    Zt = Y[maxlag-1:-1,:].copy()
    for i in range(2, m+1):
        Zt = np.concatenate([Zt, Y[maxlag-i:-i,:]], axis=1)
    Zt = np.concatenate([np.ones([T,1]) ,Zt], axis=1)
    Z = Zt.copy().T
    Y = Y[maxlag:,].T
    return Y, Z

def ic_table(Y, maxlag):

    '''
        K : number of endogenous variables
        m : number of lags per equation
        T : sample size for estimation
    '''
    nobs, K = Y.shape
    T = nobs - maxlag

    ic_table = np.zeros((maxlag, 3))

    for m in range(1, maxlag+1):
        y, z = make_reg_same_sample(Y, maxlag, m)
        _, sig = OLSvar(y, z)
        ic_table[m-1, 0] = AIC(sig, m, K, T)
        ic_table[m-1, 1] = HQ(sig, m, K, T)
        ic_table[m-1, 2] = SC(sig, m, K, T)

    table = pd.DataFrame(ic_table, columns=["AIC(m)", "HQ(m)", "SC(m)"],
                 index=np.arange(1, maxlag+1))

    # to find lag lenght by minimizing info criteria
    # maxlag = 24
    # ic_lag = ic_table(data, maxlag).idxmin(axis=0)

    return table

def make_var_regressors(Yfull, lags, det='c', det_first=True):
    ''' set up VAR regressor matrix with a constant as first regressor
        as the default
    '''
    S, n = Yfull.shape
    T = S - lags
    Zt = Yfull[lags-1:-1,:].copy()
    for i in range(2, lags+1):
        Zt = np.concatenate([Zt, Yfull[lags-i:-i,:]], axis=1)

    if det == 'c':
        if det_first:
            Zt = np.concatenate([np.ones([T,1]) ,Zt], axis=1)
        else:
            Zt = np.concatenate([Zt, np.ones([T,1])], axis=1)

    elif det == 'ct':
        const = np.ones([T,1])
        if det_first:
            Zt = np.concatenate([const, const.cumsum() ,Zt], axis=1)
        else:
            Zt = np.concatenate([Zt, const , const.cumsum()], axis=1)

    Y = Yfull[lags:,:]
    return  Y, Zt

def arfit(y, lags):
    T = len(y)
    y = y.reshape((T,1))

    # set up regressor matrix with intercept
    X = y[lags-1:-1].copy() # first lag
    for i in range(2, lags+1):
        X = np.concatenate([X, y[lags-i:-i,]], axis=1) # remaining lags
    X = np.concatenate([X, np.ones((T-lags,1))], axis=1)

    # choose same sample from y
    y = y[lags:]

    # OLS formula
    XtX = X.T @ X
    Xty = X.T @ y
    beta = solve(XtX, Xty)
    e = y - X @ beta
    sigma2 = e.T @ e / (T - lags)

    return X, y, beta, sigma2

def ar_resid_variance(y, lags):
    S = len(y)
    T = S - lags # effective sample size
    y = y.reshape((S,1))

    # set up regressor matrix with intercept
    X = y[lags-1:-1].copy() # first lag
    for i in range(2, lags+1):
        X = np.concatenate([X, y[lags-i:-i,]], axis=1) # remaining lags
    X = np.concatenate([X, np.ones((T,1))], axis=1)

    # choose same sample from y
    y = y[lags:]

    # OLS formula
    XtX = X.T @ X
    Xty = X.T @ y
    beta = solve(XtX, Xty)
    e = y - X @ beta
    sigma2 = e.T @ e / (T - lags - 1) # degrees of freedom adjustment
    #betasm = sm.OLS(y, X).fit().params

    # a small test
    # print(np.allclose(betasm, beta.flatten()), betasm.shape)
    return sigma2[0,0]

def uni_resid_stderr(data, p):
    _ , n = data.shape
    resid_sig = np.ones(n)
    for var in range(n):
        sig2 = ar_resid_variance(data[:,var], p)
        resid_sig[var] = np.sqrt(sig2)

    return resid_sig

def OLSvar(Y, Z):
    n, T = Y.shape
    k = Z.shape[0]
    p = (k - 1) / n

    ZZt = Z @ Z.T
    B = Y @ Z.T @ inv(ZZt)
    U = Y - B @ Z
    SigU = U @ U.T / T

    #rmsfe1 = 1 - np.diag(SigU) / np.diag(np.cov(diff(Y.T).T))
    return B, SigU

def OLSvar2(data, lags, det_first=True):
    Yt, Zt = make_var_regressors(data, lags, det_first=det_first)
    Y, Z = Yt.T, Zt.T
    n, T = Y.shape
    k = Z.shape[0]
    p = (k - 1) / n

    ZZt = Z @ Z.T
    B = Y @ Z.T @ inv(ZZt)
    U = Y - B @ Z
    SigU = U @ U.T / T#(T - k)

    #rmsfe1 = 1 - np.diag(SigU) / np.diag(np.cov(diff(Y.T).T))
    return B, SigU

def OLSvarfit(data, lags, det_first=True):
    ''' compute the fit of the VAR on the untruncated data'''

    Yt, Zt = make_var_regressors(data, lags, det_first=det_first)
    Y, Z = Yt.T, Zt.T
    n, T = Y.shape
    k = Z.shape[0]
    p = (k - 1) / n

    ZZt = Z @ Z.T
    B = Y @ Z.T @ inv(ZZt)
    U = Y - B @ Z
    SS = U @ U.T / (T + 1) # data.shape[0] - lags
    fit = 1 - np.diag(SS) / np.diag(np.cov(diff(data).T))
    return fit

def RMSFE1(Y, SigU, k, lags, vars=None):

    # compute one step ahead msfe relative to a random walk
    n = SigU.shape[0]
    T, _  = Y.shape
    SigU = SigU *(T - k) / (T - lags + 1)
    if vars is None:
        vars = np.arange(0, n)

    MSFE = np.diag(SigU)[vars]
    Y = Y[:,vars]
    denom = np.diag(np.cov(diff(Y).T))
    rmsfe1 =  MSFE / denom

    return rmsfe1

def vech(A):
    '''
    return elements in the lower triangular of the symmetric
    matrix A in a vector stacked by columns
    '''
    low = np.triu(A) == 0
    return A[~low]

def mvech(A):
    '''
    return elements in the lower triangular of each of the symmetric
    matrices stacked in the first dimension of A
    in a vector stacked by columns

    INPUT
    ---

        A : (T, n, n) nd_array
            T symmetric matrices of diension (n x n) each
            e.g. outer products of VAR residuals

    OUTPUT
    ---

        B : (T, n*(n+1)/2) nd_array
             the n*(n+1)/2 elements in the lower triangular of each
             element in A stacked columnwise in a vector

    '''
    T, n, _ = A.shape
    nels = int(n*(n+1)/2)

    low = np.triu(A) == 0
    return A[~low].reshape(T, nels)

def lmar(resids, exo, lags, h_max):

    '''
        Performs a series of multivariate Breuch Godfrey LM tests for residual
        autocorrelation order up to h for each lag length between 1 and h_max.

        H_0 : no autocorrelationup to order h
        H_1 : there is autocorrelationin at least one lag up to order h

    INPUT
    ---
        resids : (T, n) nd_array
            residuals from a reduced form VAR with n endogenous variables
            and T observations per equation.

        exo : (T, n) nd_array
            time series of endogenous variables from the VAR to be used as
            exogenous variables in the auxilary regressions for the LM test

        lags : int
            number of lags in the original VAR

        h_max : int
            lag order up to which to perform the tests

    OUTPUT
    ---
    results : DataFrame
        summary of the test results for the usual Breusch Godfrey test and
        the small sample adjusted F-test version proposed by Edgerton and Shukur (1999).
    '''

    # define the design matrices
    res = resids
    columns = ['h', 'LM', 'p_LM', 'df1', 'LMF', 'p_LMF', 'df1', 'df2']
    results = pd.DataFrame(np.zeros([h_max, len(columns)]), columns=columns, dtype = int)
    K = exo.shape[1]

    X = np.array([exo[t - lags : t].ravel() for t in range(lags , len(exo))])
    X  = np.concatenate([X, np.ones([len(X), 1])], axis=1)
    nobs = len(res)

    ## compute the least squares solution
    # for the restricted model
    params_r = np.linalg.lstsq(X, res, rcond=-1)[0]
    resid_r = res - np.dot(X, params_r)
    #estimator of the Covariance matrix
    cov_r = 1/(nobs) * (resid_r.T @ resid_r)

    # and the restricted models for every k up to h
    for h in range(1, h_max+1):
        #for the unrestricted model includeing the lagged residuals
        res = np.concatenate([np.zeros([h, K]), resids], axis=0)
        U_hat = np.array([res[t-h : t].ravel() for t in range(h, len(res))])

        Z = np.concatenate((X, U_hat), axis=1)
        res = res[h:]
        params = np.linalg.lstsq(Z, res, rcond=-1)[0]
        resid = res - np.dot(Z, params)
        #estimator of the Covariance matrix
        cov = 1/(nobs) * (resid.T @ resid)

        n =  X.shape[1] - K
        m =  K * h
        r = np.sqrt( (K**2 * m**2 - 4) / (K**2 + m**2 - 5) )
        q = .5 * K * m - 1
        N = nobs - n - K - m - 0.5 * (K - m + 1) # the n in this formula is missing in Jmulti and Paff R documentations
        Rr2 = 1 - (np.linalg.det(cov)/np.linalg.det(cov_r))


        df1 = K**2 * h
        df2 =  N * r - q

        F = (1 - (1 - Rr2) ** (1 / r)) / (1 - Rr2)  ** (1 / r) * (N * r - q) / (K * m)
        #Rr2 * df2 / df1

        df1 = df1
        df2 = df2.astype(int)
        F_pval = (1 - stats.f.cdf(F, df1, df2 ))

        # compute the LM teststatistic with the formula from the JMulti handbook
        cov_prod = np.linalg.inv(cov_r) @ cov
        lm = nobs * (K - np.trace(cov_prod))
        df = h * (K ** 2)

        p_lm = (1 - stats.chi2.cdf(lm, df))

        col = np.array([h, lm, p_lm, df, F, F_pval, df1, df2])
        results.iloc[h - 1,:] = col

    return results

def lmarch(resids, h_max):

    '''
        Performs a series of multivariate ARCH-LM tests for conditional
        residual heteroskedasticity of order up to h for each lag length
        h between 1 and h_max.

        H_0 : no residual ARCH effects up to to order h
        H_1 : there are ARCH effects in at least one lag up to oder h

    INPUT
    ---
        resids : (T, n) nd_array
            residuals from a reduced form VAR with n endogenous variables
            and T observations per equation.

        h_max : int
            lag order up to which to perfor the tests

    INTERMEDIATE VALUES
    ---
    K : int
        number of regressors in the auxilary regressions
    n : int
        number of endogenous variables in the original Var

    OUTPUT
    ---
    results : DataFrame
        summary of the test results for the usual Breusch Godfrey test and
        the small sample adjusted F-test version proposed by Edgerton and Shukur (1999).
    '''

    # define the design matrices
    res_sq = resids[:,:,None] * resids[:,None,:]
    Y = mvech(res_sq)  # unique elements if the "squared residuals"
    _, n = resids.shape  # n = number of equaions in theoriginal  VAR
    nobs, K = Y.shape #

    columns = ['h', 'ARCH-LM', 'p_LM', 'df1']
    results = pd.DataFrame(np.zeros([h_max, len(columns)]), columns=columns, dtype = int)

    # and the restricted models for every k up to h
    for h in range(1, h_max+1):

        # unrestricted model with adjusted lentgh
        T = nobs - h
        const = np.ones((nobs-h,1))
        params_r = solve(const.T@ const, const.T @ Y[h:])
        resid_r = Y[h:] - np.dot(const, params_r)
        # estimator of the restricted Covariance matrix
        cov_r = 1/(T) * (resid_r.T @ resid_r)
        cov_r_inv = solve(cov_r, np.eye(cov_r.shape[0]))

        # set up regressors for the auxiliary regression
        U_hat_sq = np.array([Y[t-h : t].ravel() for t in range(h, len(Y))])
        Z = np.concatenate((const, U_hat_sq), axis=1)
        # change length of the left hand side variables accordingly
        y = Y[h:]
        params = np.linalg.lstsq(Z, y, rcond=None)[0]
        u = y - np.dot(Z, params)

        # estimator of the unrestricted Covariance matrix
        cov_unr = 1 / T * (u.T @ u)

        df = int(h * K**2)
        Rmsq = 1 - (1 / K) * np.trace(cov_unr @ cov_r_inv)
        LM = T * K * Rmsq

        pval = (1 - stats.chi2.cdf(LM, df))

        col = np.array([h, LM, pval, df])
        results.iloc[h - 1,:] = col

    return results

def make_companion(params, n, lags):

    '''
    put a parameter vector into companion form

    INPUT
    ---
    params : (n, n*lags) nd_array
        redduced form parameters of of lagged endogenous variables from a VAR excluding
        any coefficients of deterministic regressors.

    n : int
        number of endogenous varables in the VAR

    lags : int
        number of lags as regressors

    OUTPUT
    ---
    c : nd_array
        companion or state space form of the reduced form VAR

    '''

    I = np.identity(n*lags-n)
    I = np.concatenate((I, np.zeros([n*lags - n,n])), axis=1)

    c = np.concatenate((params,I), axis=0)

    return c

def get_irfs(params, n, lags, periods, Sigma):

    '''
    function to compute impulse responses from reduced form parameters using
    a cholseky identification scheme


    INPUT
    ---
    params : (n, n*lags) nd_array
        redduced form parameters of of lagged endogenous variables from a VAR excluding
        any coefficients of deterministic regressors.

    n : int
        number of endogenous varables in the VAR

    lags : int
        number of lags as regressors

    periods : int
        number of periods for which to compute the impulse response coefficients

    Sigma : (n, n) nd_array
        reduced form residual covariance matrix from the VAR


    OUTPUT
    ---
    irfs_organized : (periods, n, n) nd_array
        impulse response coefficients. Axis correspond
        [periods, #impulse, #response].
        [:, 0, 2] gives the impulse response of the 3rd variable to a shock in the
        first equation.

    '''

    C = make_companion(params, n, lags)
    B = np.linalg.cholesky(Sigma)

    IRF = np.concatenate((B, np.zeros([n * lags - n, n])), axis=0)
    big_IRF = B
    for i in np.arange(1, periods):
        new_IRF = np.dot(C, IRF)
        big_IRF = np.concatenate((big_IRF, new_IRF[:n,:n]), axis=1)
        IRF = new_IRF

    irfs_organized = np.zeros((periods, n, n))
    for impulse in range(n):
        subset = list(np.arange(impulse ,periods*n , n))
        irfs_organized[:,impulse,:]=  big_IRF[:,subset].T

    return irfs_organized
