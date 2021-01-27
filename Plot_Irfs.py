import numpy as np
import os
import pandas as pd
import datetime as dt

import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt

from olsvar import *
from littBVAR import *


print('\nstarting script...\n')
# set some options
run_Gibbs = False
plot_irfs = True
irf_sample = 'full' #'post83' 'full'
savefig = True
irh = 48
ndraws = 200
lower, upper = [5, 16], [95, 84]
find_shrinkage = True
sum_of_coefs = True

data_file = r"hof.xls"

# specify variables for the different VARs
varlist_small  =[ 32, 114, 86 ]
varlist_cee = [ 32, 114, 112,  86,  72,  76,  77 ]
varlist_medium = [ 32, 114, 112, 1, 2, 5, 19, 24, 50, 108, 124,
                  128, 86, 71, 72, 76, 77, 82, 92, 103 ]
litt_det_pos = False

# set up paramaters for the impulse response analysis
if irf_sample == 'full':
    start_irf_y = '1961'
    end_irf_y = '2002'
elif irf_sample == 'pre83':
    start_irf_y = '1961'
    end_irf_y = '1983'
elif irf_sample == 'post83':
        start_irf_y = '1984'
        end_irf_y = '2002'

# parameters for optimizing the shrinkage
start_train = '02-01-1960'
end_train = '02-01-1970'

# set up the model info objects
lag13 = 13
policy_var = 86        # index of fed funds rate
maxlag = 24     # max lag to comute info criteria
small = BVARinfo(varlist=varlist_small, lags=lag13, title='SMALL', shrink=1000)
cee = BVARinfo(varlist=varlist_cee, lags=lag13, shrink=0.4659, title='CEE')
medium = BVARinfo(varlist=varlist_medium, lags=lag13, shrink=0.155, title='MEDIUM')
models = [small, cee, medium]

# prepare data
A = pd.read_excel(data_file, 0, header=0, index_col=None,
                  skiprows=[1, 2], usecols=np.arange(1, 133))
dates = pd.date_range(start='01-01-1959', end='01-01-2004', freq='M')
A.index = dates

# read transformation codes
B = pd.read_excel(data_file, 2, header=None, index_col=None, skiprows=1,
                  usecols=[1, 2, 3],
                  names=['code', 'info', 'transcode'])

# transform = 4, 5, 6 indicates taking the log
# transform = 1, 4 indicates using a random walk prior
take_logs = B.transcode.isin([4, 5, 6])

# take log for all those variables
X = A.values.copy()
X[:,take_logs] = np.log(X[:,take_logs]) * 100
X = pd.DataFrame(X, columns=A.columns, index=dates)

# transform = 1, 4 indicates using a random walk prior
RW = ~B.transcode.isin([1,  4])
RW = RW.astype(int)

# add information to the models
for mod in models:
    mod.eval_names = ['employment', 'CPI', 'Fed Funds rate']
    mod.codes = A.columns[mod.varlist] # names of variables
    mod.RW = RW[mod.varlist].values # prior mean indicator (0 or 1)
    mod.data = X[start_irf_y:end_irf_y].values[:,mod.varlist]
    mod.train_data = X[start_train:end_train].values[:,mod.varlist]
    mod.ar_std = uni_resid_stderr(mod.data, mod.lags)
    mod.means = mod.data.mean(axis=0)
    mod.vareval = [(mod.varlist[var] == np.array(varlist_small)).any() for var in range(mod.n)]
    mod.impulse_pos = [(mod.varlist[var] == policy_var) for var in range(mod.n)]
    mod.ic = ic_table(mod.data, maxlag).idxmin(axis=0)
    mod.k = mod.lags * mod.n + 1
    mod.train_ar_std = uni_resid_stderr(mod.train_data, mod.lags)
    mod.train_means = mod.train_data.mean(axis=0)

# get the OLS fit oin the trainig samlpe
training_data = X[start_train:end_train].values[:,small.varlist]
ytrain, xtrain = make_var_regressors(training_data, small.lags, det_first=False)
Btrain, Sigtrain = OLSvar(ytrain.T, xtrain.T)
train_fit = OLSvarfit(training_data, lags=small.lags, det_first=False)

# check equatlity with original program
temp_args = [0.2621, cee.RW, cee.train_ar_std, cee.train_means]
litt_Yd, littXd = Litt_dummies(cee.n, cee.lags, temp_args, det_first=False, sum_of_coefs=sum_of_coefs)
Blitt, Siglitt, littfit = BVAR_litt2(cee.train_data, cee.lags, temp_args,
                      det_first=False, sum_of_coefs=sum_of_coefs)
littfit = littfit[cee.vareval].mean()
# fit the litterman var with the shrinkage parameter from Banbura et al.


# find lambda to get the closest to the OLS fit
if find_shrinkage:
    print('=============================')
    print('\n DETERMINING  SHRINKAGE:')
    print('\n prior on sum of coefficients:\t', sum_of_coefs)
    print('\n benchmark fit:\t', '%.4f' % train_fit.mean())
    Grid = np.concatenate([np.arange(1e-10,5.01,0.025), np.ones(1)*50])
    for mod in [cee, medium]:
        Grid_vals = Grid * np.sqrt( mod.n * mod.lags)
        # y, x = make_var_regressors(mod.train_data, mod.lags, det_first=False)
        FIT = np.zeros(len(Grid_vals))
        for ii,pi in enumerate(Grid_vals):
            shrink = 1/pi
            temp_args = [shrink, mod.RW, mod.train_ar_std, mod.train_means]
            _, _, fit = BVAR_litt2(mod.train_data, mod.lags, temp_args,
                                  det_first=False, sum_of_coefs=sum_of_coefs)
            FIT[ii] = fit[mod.vareval].mean()
        fit_diff = np.abs(FIT - train_fit.mean())
        Jstar = fit_diff.argmin()
        mod.shrink = 1 / Grid_vals[Jstar]
        mod.fit = FIT[Jstar]
        print('\n', mod.title, '\n shrinkage:\t', '%.4f' % mod.shrink,
              '\n fit:\t\t', '%.4f' % mod.fit)
    print('\n=============================')

#print(mod.title +' trainigs fit:\n', testfit,'\n' ,beta_test[:5,:5])
# conpute the benchmark OLS models
IRFS_ols = []
ols_imp_std = []
for ols_mod in models:
    # ols_irf_data = X[start_irf_y:end_irf_y].values[ols_mod.varlist]
    yols, xols = make_var_regressors(ols_mod.data, ols_mod.lags, det_first=False)
    B_ols, Sig_ols = OLSvar(yols.T, xols.T)
    ols_mod.resid_covar_ols = Sig_ols
    ols_mod.impulse_std_ols = np.diag(la.cholesky(Sig_ols))[ols_mod.impulse_pos]
    irfs_ols = get_irfs(B_ols[:,:-1], ols_mod.n, ols_mod.lags, irh, Sig_ols)
    olsfit = RMSFE1(yols, Sig_ols, vars=ols_mod.vareval, k=ols_mod.k, lags=ols_mod.lags)
    IRFS_ols.append(irfs_ols[:,ols_mod.impulse_pos,ols_mod.vareval])

# compute one BVAR model
#models = [cee, medium]
bands_litt = []
IRFS_litt = []
orig_shrink = [1000, 0.4656, 0.155]
models = [small, cee, medium]
print('\n in sample fit:')
for ii,mod in enumerate(models):
    lam = mod.shrink#orig_shrink[ii]
    temp_args = [lam, mod.RW, mod.ar_std, mod.means]

    #y, x = make_var_regressors(mod.data, mod.lags, det_first=False)
    mod.params, mod.sigma, fit = BVAR_litt2(mod.data, mod.lags, temp_args,
                                 det_first=False, sum_of_coefs=sum_of_coefs)

    mir, ub , lb = BVAR_irf(mod.data, mod.lags, temp_args, det_first=False, sum_of_coefs=sum_of_coefs,
                            irh=irh, ndraws=ndraws, upper=upper, lower=lower)
    # bands_litt.append(bandi)
    # scale impuls responses to have a unit impact on the fed funds rate
    mod.impulse_std = np.diag(la.cholesky(mod.sigma))[mod.impulse_pos]

    mod.rel_fit = fit[mod.vareval].mean()#RMSFE1(y, mod.sigma, vars=mod.vareval, lags=mod.lags, k=mod.k)

    print('\n ' + mod.title + '\t', '%.4f' % fit[mod.vareval].mean())

    # retrieve impuse responses
    irfs_litt = get_irfs(mod.params[:-1,:].T, mod.n, mod.lags,
                         irh, mod.sigma)
    mod.irf = mir[:,mod.impulse_pos, mod.vareval]
    mod.ub = ub[:,:,mod.impulse_pos, mod.vareval]
    mod.lb = lb[:,:,mod.impulse_pos, mod.vareval]
    # choose the policy impulse and thee rsponses to plot
    IRFS_litt.append(irfs_litt[:,mod.impulse_pos, mod.vareval])
print('\n=============================')

# plot responses to a monetary policy shock
if plot_irfs:
    fig_name = 'IrfPlot_' + irf_sample
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ylims = [(-1.25 , 0.4), (-1.5, 0.4), (-.5,1.5) ]
    band_names = ['$90\%$', '$68\%$']
    tickmarks = np.arange(0,irh+1,12)
    lines = [':', '--', '>g']

    fig, ax = plt.subplots(small.n,small.n, figsize=(8, 8))
    for row in range(small.n): # iterate over variables
        for ii,mod in enumerate(models):    # iterate over models

            birf = mod.irf #/ mod.impulse_std
            oirf = IRFS_ols[ii] / mod.impulse_std_ols
            ax[row,ii].plot(birf[:,row], '--k',alpha=0.7, label='BVAR')

            if mod.title in ['SMALL', 'CEE']:
                ax[row,ii].plot(oirf[:,row], '-r', alpha=0.6, label='OLS')
            elif irf_sample == 'full':
                ax[row,ii].plot(oirf[:,row], '-r', alpha=0.6, label='OLS')

            ax[row,ii].plot(np.zeros(48), ':k', lw=1)

            for bb in range(len(lower)):
                upper_band = mod.ub[bb,:,row] #/# mod.impulse_std
                lower_band = mod.lb[bb,:,row] #/ #mod.impulse_std
                #ax[row,ii].plot(upper_band, '-r', lw=.5)# label='{}'.format(upper[bb]))
                #ax[row,ii].plot(lower_band, '-r', lw=.5)# label='{}'.format(lower[bb]))
                ax[row,ii].fill_between(np.arange(48),upper_band,lower_band,
                                        color='black', alpha=1/((bb+1)*2.5),
                                        label=band_names[bb])
            if not irf_sample == 'post83':
                ax[row,ii].set_ylim(ylims[row])
            ax[row,ii].set_xticks(tickmarks)

            if row == 2:
                ax[row,ii].set_ylim((-1,1.5))
                tit = 'std. MP-shock:'
                sigb ='\nBVAR$=%.3f$' % mod.impulse_std[0]
                sigo = '\nOLS$=%.3f$' % mod.impulse_std_ols[0]
                std_text = tit + sigb + sigo
                if mod.title == 'SMALL':
                    std_text = tit + sigo
                if mod.title == 'MEDIUM' and irf_sample == 'full':
                    std_txt = tit + sigb + sigo
                elif mod.title == 'MEDIUM':
                    std_text = tit + sigb

                ax[row,ii].text(48, 1.4, std_text, fontsize=10,
                                verticalalignment='top', horizontalalignment='right')
        #ax[col].plot(lower[:,2,col], ':k')
        #ax[col].plot(upper[:,2,col], ':k')
            if ii == 0:
                ax[row,ii].set_ylabel(mod.eval_names[row])
            if row == 0:
                ax[row,ii].set_title(mod.title)
    ax[2, 1].legend(loc='upper center', bbox_to_anchor=(0.4, -0.05), ncol=4,
               frameon=False)
    plt.tight_layout()
    if savefig:
        if sum_of_coefs:
            fig.savefig(fig_name + '.pdf')
        elif not sum_of_coefs:
            fig.savefig(fig_name + '_without_sum_of_coefs.pdf')
    plt.show()

if run_Gibbs:
    prior_args = [small.shrink, small.RW, small.ar_std, small.means]
    Yd, Xd = Litt_dummies(small.n, small.lags, prior_args, sum_of_coefs=sum_of_coefs)
    # compute prior based on the Litt_dummies
    Omega0 = np.kron(np.diag(small.ar_std), Xd.T @ Xd)
    B0 = la.solve(Xd.T @ Xd, Xd.T @ Yd)
    e = Yd - Xd @ B0
    S0 = e.T @ e
    k = (small.lags * small.n + 1)
    a0 = Yd.shape[0] - k
    #B0 = np.concatenate([B0[-1,:].reshape(1, mod.n), B0[:-1,:]], axis=0)
    gibbs_args = (a0, S0, B0.reshape((small.n* k,1), order='F'), Omega0)
    betamat, sigmat = BVAR_Gibbs(small.data, small.lags, nsim=2000, burnin=1000,
                                 prior_args=gibbs_args, print_step=500)
    betamean = betamat.mean(axis=0).reshape(small.n, k)

    print('testing parameter:',
          np.allclose(betamean.T[1:,:], small.params[:-1,:], atol=1e-2))
    # the tolerance naturally depends on the chosen number of replications

print('\n...script finished!')
