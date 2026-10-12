# LargeBvarPython

Python implementation of the Large Bayesian VAR of Bańbura, Giannone & Reichlin (2010, *Journal of Applied Econometrics*), written from scratch with NumPy. The code replicates the paper's impulse responses to a monetary policy shock for the SMALL, CEE and MEDIUM specifications and compares them to OLS estimates.

![Impulse responses: BVAR vs. OLS](IrfPlot_full.pdf)

*Responses of employment, CPI and the Federal Funds rate to a monetary policy shock. Dashed: BVAR (posterior median) with 68% / 90% bands. Red: OLS.*

## What the code does
- **olsvar.py**: VAR estimation by OLS, including impulse responses
- **littBVAR.py**: Large BVAR with a Litterman/Minnesota-type prior implemented via dummy observations
- **Plot_Irfs.py**: selects the shrinkage parameter as in Bańbura et al., estimates the three specifications and produces the figure above

## Why it is interesting
With many variables, OLS impulse responses become erratic (see the MEDIUM column). Bayesian shrinkage keeps them economically plausible, which is the paper's main point.

## Context
Study project from the start of my PhD (2017/18). I later moved my research to MATLAB, so the code is not actively maintained.
