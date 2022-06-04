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