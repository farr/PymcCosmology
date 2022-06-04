import aesara.tensor as at
import numpy as np
import pymc as pm

from .cosmology import Ez, dCs, dLs
from .utils import interp

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
        dVCinterp = 4*np.pi*dCinterp*dCinterp*dH/Ez(zinterp, Om, w)

        zden = (1+zinterp)**1.7 / (1 + ((1+zinterp)/(1+1.9))**5.6) * dVCinterp
        norm = 0.5*at.sum((zinterp[1:] - zinterp[:-1])*(zden[1:] + zden[:-1]))
        log_norm = at.log(norm)

        z = pm.Uniform('z', 0, zmax, initval=abs(zs_obs), shape=zs_obs.shape[0])
        pm.Potential('zprior', at.sum(1.7*at.log1p(z) - at.log1p(((1+z)/(1+1.9))**5.6) + at.log(interp(z, zinterp, dVCinterp)) - log_norm)) # SFR / (1+z)

        dL = pm.Deterministic('dL', interp(z, zinterp, dLinterp))

        pm.Normal('z_likelihood', mu=z, sigma=sigma_zs_obs, observed=zs_obs)
        pm.Normal('dL_likelihood', mu=dL, sigma=sigma_dls_obs, observed=dls_obs)
    return model