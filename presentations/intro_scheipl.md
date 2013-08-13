What am I?
========================================================

- PostDoc @ LMU Munich
- working on additive mixed models for correlated functional data  
- very grateful for the invitation

Why am I here?
========================================================

- author of `RLRsim`: exact (restricted) likelihood ratio tests for variance components in `lme4` or `nlme`: 
    - finite sample distribution of likelihood ratio is a function of standard normal variates and the eigenvalues of a matrix derived from ${\bm X, \bm Z}$.

- author of `amer` (replaced by `gamm4`) for *additive* mixed models with `lme4`:
    - quadratic penalty on spline coefficients mathematically equivalent to (partially improper) Gaussian prior $\Rightarrow$ reparameterize & estimate as MEM. 

Also:

- author of `spikeSlabGAM`: Bayesian model averaging/model selection for GAMMs

What have I been doing recently?
========================================================

`refund::pffr()`:

- is a wrapper for `mgcv`/`gamm4` that fits functional *additive* mixed models:
- `pffr(Y ~ s(z) + ff(X, yind=t, xind=s) + s(id, bs='re'))`
   
    fits the model 
   
    $y_{ij}(t) = \beta_0(t) + f(z_{ij}, t) + \int x_{ij}(s) \beta(s,t) ds + b_i(t) + 
\epsilon_{ij}(t)$
- more: spatial effects, FPC-based function-on-function effects, GLS-type estimators with 
${\hat{\text{Cov}}(\epsilon)}$, ...


What am I working on presently?
========================================================

with Torsten Hothorn: `tikhonov` 

- a (C++) framework for efficiently representing and computing with potentially *very* large 
model matrices 
- basically an extension of Currie/Durban/Eilers "Generalized Linear Array Models"  with tricks from Stefan Lang's `BayesX`
- model matrix ${\bm M = (\bm M_1 | \bm M_2 | \dots  | \bm M_p)}$
- ${\bm M_j = \bm S_j \bm D_j \bm C_j}$ with sampling matrix ${\bm S_j}$, design matrix ${\bm D_j}$, contrast/penalty ${\bm C_j}$ 
- ${\bm S_j \sim n \times d_j}$ just an index: which observation has which value, i.e. binning of 
continuous covariates into $d_j$ bins. Also: sorting to reduce infill, etc.
- ${\bm C_j}$ enforces constraints for identifiability and/or absorbs penalty into design 

`tikhonov`
========================================================

- provide methods for computing, for different types & combinations of ${\bm M_i, \bm M_j}$:
    - weighted inner products ${\bm  M_i^T  \text{diag}(\bm w) \bm M_j}$,
    - linear functions ${\bm M_i  \bm\theta}$, ${ \bm M_i^T \bm\theta}$,
    - diagonals of quadratic forms ${\text{diag}(\bm M_i \bm \Sigma_{ij}  \bm M_j^T)}$

- that's pretty much all you need for (P-)IRLS, (G)LS.
- still very early stages

What can I contribute?
========================================================

@ **Matrix representations**:

- some experience from working on `tikhonov`

@ **Automatic detection of ill-defined models**:

- copy some of the functionality of `mgcv::gam.side()`? 

What can I contribute?
========================================================

@ **Inference and model selection**:

- improvements to `RLRsim`:

    - no more error-prone refitting of partial models
    - maybe: integration into `summary.merMod` (as an option)?   
- discuss: Simon Wood's recently proposed $F$-type tests for splines
- implementation/notation for dependent random effects ${ \bm b \sim N_q( \bm 0, \tau^2 \bm D)}$ where ${\bm D}$ is fixed *a priori* (genetic/spatial proximity etc.):
   
    - can this be done efficiently? (preserve sparsity, etc.)

What can I contribute?
========================================================

@ **User interfaces**:

- more compact formula notation for un-correlated or $i.i.d.$ random effects:
   - for factor `f`, `(0 + f | subject)` yields correlations between intercepts for levels of `f`
   - could we write s.th. like `(0 + f || subject)` or `(1 + x || subject)` to get un-correlated random effects?
   - could we write s.th. like `(0 + f ||| subject)` to enforce identical variances for the random intercepts for the different levels?  


What else?
========================================================

- speed vs. stability of new `glmer`: rumour has it it's (much) slower, has it become better? what about scale parameter estimates?   
- in that vein: should 
`http://glmm.wikidot.com/pkg-comparison`
also link to some performance comparisons?  

<!---
pandoc -t beamer --template pandocBeamerTemplate.tex intro.md -o intro.pdf
--->
