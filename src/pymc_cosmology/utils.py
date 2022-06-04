import aesara.tensor as at
import aesara.tensor.extra_ops as ate

def interp(x, xs, ys):
    """Aesara op to linearly interpolate the function whose values are `ys` at the points `xs` to the points `x`."""
    x = at.as_tensor(x)
    xs = at.as_tensor(xs)
    ys = at.as_tensor(ys)

    N = xs.shape[0]

    ind = ate.searchsorted(xs, x)

    # Clamp the index in [1, N); this will have the effect of extending the function to be interpolated to be linear outside the defined range of x
    ind = at.where(ind < 1, 1, ind)
    ind = at.where(ind >= N, N-1, ind)

    r = (x - xs[ind-1])/(xs[ind]-xs[ind-1])
    return r*ys[ind] + (1-r)*ys[ind-1]

def md_sfr(z, a, z_p, c):
    r"""Returns a parameterized version of the [Madau & Dickinson](https://arxiv.org/abs/1403.0007) SFR.
    
    .. math::
        r(z) = \frac{\left(1 + z\right)^a}{1 + \left( \frac{1+z}{1+z_p}\right)^c}
    """
    opz = 1 + z
    return opz**a / (1 + (opz/(1+z_p))**c)

def trapz(ys, xs):
    """Aesara implementation equivalent to `np.trapz` (note argument order)!"""
    return 0.5*at.sum((xs[1:] - xs[:-1])*(ys[1:] + ys[:-1]))

def cumtrapz(ys, xs):
    """Aesara implementation equivalent to `cumtrapz` with initial 0 output."""
    integrand = 0.5*(xs[1:] - xs[:-1])*(ys[:-1] + ys[1:])
    return at.concatenate([at.as_tensor([0.0]), at.cumsum(integrand)])