import aesara.tensor as at
import numpy as np
from .utils import cumtrapz

def Ez(z, Om, w):
    """Aesara definition of the cosmological integrand for a flat wCDM cosmology.
    
    See [Hogg (1999)](https://arxiv.org/abs/astro-ph/9905116).  

    Parameters
    ----------
    Om : real 
        Present day matter density relative to the critical density `0 <= Om <= 1`.
    w : real
        Dark energy equation of state parameter.
    """
    opz = 1 + z
    return at.sqrt(Om*opz*opz*opz + (1-Om)*opz**(3*(1+w)))

def dCs(zs, Om, w):
    """Evaluate the (unitless) comoving distance integral on the grid `zs`.
    
    See [Hogg (1999)](https://arxiv.org/abs/astro-ph/9905116).
    
    Parameters
    ----------
    zs : real array
        The redshifts at which the comoving distance should be calculated. These
        must be sufficiently dense that a trapezoidal approximation to the
        integral is sufficiently accurate.  A common choice is to choose them
        uniformly in `log(1+z)` via `zs = np.expm1(np.linspace(np.log(1),
        np.log(1+zmax), Nz))`
    Om : real
        Dimensionless matter density at the present day.
    w : real
        Dark energy equation of state parameter.    
    """
    fz = 1/Ez(zs, Om, w)
    return cumtrapz(fz, zs)

def dLs(zs, dCs):
    """Luminosity distance on the grid `zs` with previously evaluated comoving distances.

    See [Hogg (1999)](https://arxiv.org/abs/astro-ph/9905116).

    Parameters
    ----------
    zs : real array
        Grid of redshifts.  See `pymc_cosmology.cosmology.dCs`.
    dCs : real array
        Comoving distances on the grid of redshifts.
    """
    return dCs*(1+zs)

def dVdz(zs, dCs, Om, w):
    """Evaluate the (unitless) differential comoving volume on the grid `zs`.
    
    See [Hogg (1999)](https://arxiv.org/abs/astro-ph/9905116).

    Parameters
    ----------
    zs : real array
        The redshifts at which the comoving distance should be calculated. These
        must be sufficiently dense that a trapezoidal approximation to the
        integral is sufficiently accurate.  A common choice is to choose them
        uniformly in `log(1+z)` via `zs = np.expm1(np.linspace(np.log(1),
        np.log(1+zmax), Nz))`
    dCs : real array
        The (unitless) comoving distances at the redshift grid.
    Om : real
        Dimensionless matter density at the present day.
    w : real
        Dark energy equation of state parameter.  
    """
    return 4*np.pi*dCs*dCs/Ez(zs, Om, w)