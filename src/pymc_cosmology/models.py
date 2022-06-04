import aesara.tensor as at
import numpy as np
import pymc as pm

from .cosmology import Ez, dCs, dLs, dVdz
from .utils import interp, md_sfr, trapz

def make_redshift_distance_gaussian_model(zs_obs, sigma_zs_obs, dls_obs, sigma_dls_obs, zmax=100, Nz=1024):
    r"""A model fitting Gaussian uncertainty measurements of redshift and
        distance.
    
    The model assumes that the true redshifts are distributed according to the
    [Madau & Dickinson (2014)](https://arxiv.org/abs/1403.0007) SFR.  The priors
    imposed are approximately flat in `0.35 < h < 1.4`, `0 < Om < 1`, and `-1.5
    < w < -0.5`.  The code stores some auxillary variables in the sampling
    chain, too: 

    * `dH` : the Hubble distance.
    * `Ode` : the dark energy density at the present day (`= 1 - Om`).

    It also stores the inferred true `z` and `dL` (Gpc) for each observation
    based on the cosmology and the observed redshift and distance and their
    uncertainties.

    The model assumes that the quoted uncertainties correspond to the standard
    deviation of a Gaussian likelihood for each redshift and distance
    measurement (and that the measurements are independent of each other).

    Parameters
    ----------
    zs_obs : real array
        Observed redshifts.
    sigma_zs_obs : real array
        Uncertainty (Gaussian s.d.) on the observed redshifts.
    dls_obs : real array
        Observed luminosity distances (in Gpc).
    sigma_dls_obs : real array
        Uncertainty (Gaussian s.d.) on the observed luminosity distances.
    zmax=100 : real
        Maximum redshift for the interpolated cosmology grid.
    Nz=1024 : int
        Number of points in redshift grid (uniformly distributed in `log(1+z)`).

    Returns
    -------
    A pymc model suitable for inference of cosmology and true redshifts /
    distances from the observed data.

    Implementation Notes
    --------------------
    We sample in a slightly funny parameterization: I have found that the
    best-constrained :math:`h(z)` is around :math:`z \sim 0.5` when fitting a
    universe full of objects (probably this due to the transition from dark
    energy dominated to matter dominated around this point, which gives nice
    structure that the sampler can latch onto).  So we sample in :math:`\left(
    h(0.5), \Omega_M, w \right)`.  We want, however, to impose a flat prior in
    :math:`\left(h, \Omega_M, w\right)`, so a Jacobian adjustment is necessary.
    Other than this, the code is pretty straightforward.

    """
    zinterp = np.expm1(np.linspace(np.log(1), np.log1p(zmax), Nz))

    with pm.Model() as model:
        w = pm.Uniform('w', -1.5, -0.5, initval=-1)

        # We transform from h to h(0.5), which is (about) the best-constrained
        # redshift for h.  h(0.5) = h*Ez(0.5), and we incorporate a Jacobian
        # factor so that the prior density is uniform in h.
        hz0p5 = pm.Uniform('hz0.5', 0.5, 2, initval=1)
        Om = pm.Uniform('Om', 0, 1, initval=0.3)
        Ezfac = Ez(0.5, Om, w)
        h = pm.Deterministic('h', hz0p5 / Ezfac)
        Ode = pm.Deterministic('Ode', 1-Om)
        om = pm.Deterministic('om', Om*h*h)
        ode = pm.Deterministic('ode', Ode*h*h)
        pm.Potential('h_hz2_jacobian', -at.log(Ezfac))
        
        dH = pm.Deterministic('dH', 2.99792 / h) # Gpc

        dCinterp = dH*dCs(zinterp, Om, w)
        dLinterp = dLs(zinterp, dCinterp)
        dVCinterp = dH*dH*dH*dVdz(zinterp, dCinterp, Om, w)

        zden = md_sfr(zinterp, 2.7, 1.9, 5.6)*dVCinterp/(1+zinterp)
        norm = trapz(zden, zinterp)
        log_norm = at.log(norm)

        z = pm.Uniform('z', 0, zmax, initval=abs(zs_obs), shape=zs_obs.shape[0])
        pm.Potential('zprior', at.sum(at.log(interp(z, zinterp, zden)) - log_norm))

        dL = pm.Deterministic('dL', interp(z, zinterp, dLinterp))

        pm.Normal('z_likelihood', mu=z, sigma=sigma_zs_obs, observed=zs_obs)
        pm.Normal('dL_likelihood', mu=dL, sigma=sigma_dls_obs, observed=dls_obs)
    return model

def make_chirp_mass_mf_cosmology_model(mc_dets, log_dls_obs, sigma_log_dls_obs, zmax=100, Nz=1024):
    r"""Mass function cosmology model based on chirp mass and distance
    measurements.

    This model fits a Gaussian population (mean and s.d. are parameters) to the
    observed chirp masses (assumed to be measured without uncertainty, since it
    is appropriate for neutron stars where the chirp mass is measured to ppt
    uncertainty or better) and a parameterized [Madau & Dickinson
    (2014)](https://arxiv.org/abs/1403.0007) merger rate to observations of
    luminosity distance (assumed log-normally distributed).

    Parameters
    ----------
    mc_dets : real array
        The detector-frame chirp masses (assumed to be measured without
        uncertainty).
    log_dls_obs : real array
        The log of the observed luminosity distances (in Gpc).
    sigma_log_dls_obs : real array
        The uncertainty on the observed log dLs.
    zmax=100 : real
        The maximum redshift.
    Nz=1024 : int
        The number of steps (uniform in `log(1+z)`) in the redshift grid for
        interpolating cosmology.

    Returns
    -------
    A pymc model for this measurement.  Parameters stored in the chain include 

    mc_mc : real 
        The population mean of the chirp mass (source frame).
    sigma_mc : real 
        The population s.d. of the chirp mass (source frame).
    a : real 
        The low-redshift M-D evolution exponent.
    z_p : real 
        The (approximate) peak redshift in the M-D merger rate.
    c : real 
        The exponent of `1+z` in the M-D denominator.
    h : real 
        Dimensionless Hubble parameter.
    Om : real 
        Dimensionless matter density
    w : real 
        Dark energy equation of state.
    Ode : real 
        Dimensionless dark energy density.
    dH : real
        Hubble distance (Gpc).
    z : real array 
        The inferred true redshifts of each source.
    dL : real array
        The inferred true luminosity distance to each source (Gpc).    

    Implementation Notes
    --------------------
    In addition to assuming that the chirp mass measurements presented are
    perfectly accurate, and that the luminosity distance measurements have a
    log-normal distribution, the model is written in such a way that it samples
    much more efficiently when the uncertainties on the dL measurements are very
    small (i.e. the posterior is dominated by the dL likelihood function, not
    the M-D redshift prior).

    We also sample in the *physical* matter densities (`om = Om*h*h`) and dark
    energy densities instead of `h` and `Om`, because these parameters are very
    correlated (this is the combination that appears in the integrand for the
    comoving distance).
    """
    mc_dets = np.atleast_1d(mc_dets)
    nobs = mc_dets.shape[0]

    zinterp = np.expm1(np.linspace(np.log(1), np.log1p(zmax), Nz))

    with pm.Model() as model:
        mu_mc = pm.Normal('mu_mc', 1.2, 0.5)
        sigma_mc = pm.HalfNormal('sigma_mc', 0.15)

        a = pm.Bound('a', pm.Normal.dist(2.7, 0.5), lower=1.7, upper=3.7)
        z_p = pm.Bound('z_p', pm.Normal.dist(1.9, 0.5), lower=0.9, upper=2.9)
        c = pm.Bound('c', pm.Normal.dist(5.6, 0.5), lower=4.6, upper=6.6)

        om = pm.LogNormal('om', np.log(0.7*0.7*0.3), 0.1)
        ode = pm.LogNormal('ode', np.log(0.7*0.7*0.7), 0.1)
        h = pm.Deterministic('h', at.sqrt(om+ode))
        Om = pm.Deterministic('Om', om/(h*h))
        w = pm.Normal('w', -1, 0.2)
        Ode = pm.Deterministic('Ode', 1-Om)
        dH = pm.Deterministic('dH', 2.99792 / h) # Gpc

        dCinterp = dCs(zinterp, Om, w)
        dLinterp = dLs(zinterp, dCinterp)
        dVinterp = dVdz(zinterp, dCinterp, Om, w)

        dCinterp = dH*dCinterp
        dLinterp = dH*dLinterp
        dVinterp = dH*dH*dH*dVinterp

        ddLdzinterp = dCinterp + dH*(1+zinterp)/Ez(zinterp, Om, w)
        zdeninterp = md_sfr(zinterp, a, z_p, c)*dVinterp/(1+zinterp)
        log_znorm = at.log(trapz(zdeninterp, zinterp))

        log_d_unit = pm.Flat('log_dl_unit', shape=nobs)
        dL = pm.Deterministic('dL', at.exp(log_dls_obs + sigma_log_dls_obs*log_d_unit))
        z = pm.Deterministic('z', interp(dL, dLinterp, zinterp))

        # z follows M-D SFR; since we sample in `log_dl_unit`, we need p(z) d(z)/d(dL) d(dL)/d(log_dl_unit) = p(z) / d(dL)/dz d(dL)/d(log_dl_unit) = p(z) / d(dL)/dz * sigma_log_dl*dL
        pm.Potential('zprior', at.sum(at.log(interp(z, zinterp, zdeninterp)) - log_znorm))
        pm.Potential('zjac', at.sum(-at.log(interp(z, zinterp, ddLdzinterp) + at.log(dL) + at.log(sigma_log_dls_obs))))

        mc = pm.Deterministic('mc', mc_dets / (1 + z))
        pm.Potential('mcprior', at.sum(pm.logp(pm.Normal.dist(mu_mc, sigma_mc), mc)))
        pm.Potential('mcjac', at.sum(-at.log1p(z))) # Comes from integrating over delta-function likelihood for mc

        pm.Normal('dl_likelihood', at.log(dL), sigma_log_dls_obs, observed=log_dls_obs)
    return model
