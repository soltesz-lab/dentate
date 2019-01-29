import sys, math, numbers, itertools
import numpy as np
import scipy

def gaussian_kde(x, y, bin_size, **kwargs):
    """Kernel Density Estimation with Scipy"""

    data  = np.vstack([x, y])
    kde   = scipy.stats.gaussian_kde(data, **kwargs)

    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)

    dx = int((x_max - x_min) / bin_size)
    dy = int((y_max - x_min) / bin_size)

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, dx), \
                         np.linspace(y_min, y_max, dy))

    data_grid = np.vstack([xx.ravel(), yy.ravel()])
    z    = kde.evaluate(data_grid)
    
    return xx, yy, np.reshape(z, xx.shape)

### Kernel class definition based on spykeutils library by Robert Pröpper



class Kernel(object):
    """ Base class for kernels.  """

    def __init__(self, kernel_size, normalize):
        """
        :param kernel_size: Parameter controlling the kernel size.
        :type kernel_size: Quantity 1D
        :param bool normalize: Whether to normalize the kernel to unit area.
        """
        self.kernel_size = kernel_size
        self.normalize = normalize

    def __call__(self, t, kernel_size=None):
        """ Evaluates the kernel at all time points in the array `t`.
        :param t: Time points to evaluate the kernel at.
        :type t: Quantity 1D
        :param kernel_size: If not `None` this overwrites the kernel size of
            the `Kernel` instance.
        :type kernel_size: Quantity scalar
        :returns: The result of the kernel evaluations.
        :rtype: Quantity 1D
        """

        if kernel_size is None:
            kernel_size = self.kernel_size

        if self.normalize:
            normalization = self.normalization_factor(kernel_size)
        else:
            normalization = 1.0 * pq.dimensionless

        return self._evaluate(t, kernel_size) * normalization

    def _evaluate(self, t, kernel_size):
        """ Evaluates the kernel.
        :param t: Time points to evaluate the kernel at.
        :type t: Quantity 1D
        :param kernel_size: Controls the width of the kernel.
        :type kernel_size: Quantity scalar
        :returns: The result of the kernel evaluations.
        :rtype: Quantity 1D
        """
        raise NotImplementedError()

    def normalization_factor(self, kernel_size):
        """ Returns the factor needed to normalize the kernel to unit area.
        :param kernel_size: Controls the width of the kernel.
        :type kernel_size: Quantity scalar
        :returns: Factor to normalize the kernel to unit width.
        :rtype: Quantity scalar
        """
        raise NotImplementedError()

    def boundary_enclosing_at_least(self, fraction):
        """ Calculates the boundary :math:`b` so that the integral from
        :math:`-b` to :math:`b` encloses at least a certain fraction of the
        integral over the complete kernel.
        :param float fraction: Fraction of the whole area which at least has to
            be enclosed.
        :returns: boundary
        :rtype: Quantity scalar
        """
        raise NotImplementedError()

    def is_symmetric(self):
        """ Should return `True` if the kernel is symmetric. """
        return False


class KernelFromFunction(Kernel):
    """ Creates a kernel form a function. Please note, that not all methods for
    such a kernel are implemented.
    """

    def __init__(self, kernel_func, kernel_size):
        Kernel.__init__(self, kernel_size, normalize=False)
        self._evaluate = kernel_func

    def is_symmetric(self):
        return False


def as_kernel_of_size(obj, kernel_size):
    """ Returns a kernel of desired size.
    :param obj: Either an existing kernel or a kernel function. A kernel
        function takes two arguments. First a `Quantity 1D` of evaluation time
        points and second a kernel size.
    :type obj: Kernel or func
    :param kernel_size: Desired size of the kernel.
    :type kernel_size: Quantity 1D
    :returns: A :class:`Kernel` with the desired kernel size. If `obj` is
        already a :class:`Kernel` instance, a shallow copy of this instance with
        changed kernel size will be returned. If `obj` is a function it will be
        wrapped in a :class:`Kernel` instance.
    :rtype: :class:`Kernel`
    """

    if isinstance(obj, Kernel):
        obj = copy.copy(obj)
        obj.kernel_size = kernel_size
    else:
        obj = KernelFromFunction(obj, kernel_size)
    return obj


class SymmetricKernel(Kernel):
    """ Base class for symmetric kernels. """

    def __init__(self, kernel_size, normalize):
        """
        :param kernel_size: Parameter controlling the kernel size.
        :type kernel_size: Quantity 1D
        :param bool normalize: Whether to normalize the kernel to unit area.
        """
        Kernel.__init__(self, kernel_size, normalize)

    def is_symmetric(self):
        return True

    
class GaussianKernel(SymmetricKernel):
    r""" Unnormalized: :math:`K(t) = \exp(-\frac{t^2}{2 \sigma^2})` with kernel
    size :math:`\sigma` (corresponds to the standard deviation of a Gaussian
    distribution).
    Normalized to unit area: :math:`K'(t) = \frac{1}{\sigma \sqrt{2 \pi}} K(t)`
    """

    @staticmethod
    def evaluate(t, kernel_size):
        return np.exp(-0.5 * (t / kernel_size).simplified ** 2)

    def _evaluate(self, t, kernel_size):
        return self.evaluate(t, kernel_size)

    def normalization_factor(self, kernel_size):
        return 1.0 / (np.sqrt(2.0) * kernel_size)

    def __init__(self, kernel_size=1.0, normalize=True):
        Kernel.__init__(self, kernel_size, normalize)

    def boundary_enclosing_at_least(self, fraction):
        return self.kernel_size * np.sqrt(2.0) * \
          scipy.special.erfinv(fraction + scipy.special.erf(0.0))



def discretize_kernel(kernel, sampling_rate, area_fraction=default_kernel_area_fraction,
        num_bins=None, ensure_unit_area=False):
    """ Discretizes a kernel.
    :param kernel: The kernel or kernel function. If a kernel function is used
        it should take exactly one 1-D array as argument.
    :type kernel: :class:`Kernel` or function
    :param float area_fraction: Fraction between 0 and 1 (exclusive)
        of the integral of the kernel which will be at least covered by the
        discretization. Will be ignored if `num_bins` is not `None`. If
        `area_fraction` is used, the kernel has to provide a method
        :meth:`boundary_enclosing_at_least` (see
        :meth:`.Kernel.boundary_enclosing_at_least`).
    :param sampling_rate: Sampling rate for the discretization. The unit will
        typically be a frequency unit.
    :type sampling_rate: Quantity scalar
    :param int num_bins: Number of bins to use for the discretization.
    :param bool ensure_unit_area: If `True`, the area of the discretized
        kernel will be normalized to 1.0.
    :rtype: Quantity 1D
    """

    t_step = 1.0 / sampling_rate

    if num_bins is not None:
        start = -num_bins/2
        stop = num_bins/2
    elif area_fraction is not None:
        boundary = kernel.boundary_enclosing_at_least(area_fraction)
        if hasattr(boundary, 'rescale'):
            boundary = boundary.rescale(t_step.units)
        start = sp.ceil(-boundary / t_step)
        stop = sp.floor(boundary / t_step) + 1
    else:
        raise ValueError(
            "One of area_fraction and num_bins must not be None.")

    k = kernel(np.arange(start, stop) * t_step)
    if ensure_unit_area:
        k /= np.sum(k) * t_step
    return k


def smooth(binned, kernel, sampling_rate, mode='same', **kernel_discretization_params):
    """ Smoothes a binned representation (e.g. of a spike train) by convolving
    with a kernel.
    :param binned: Bin array to smooth.
    :type binned: 1-D array
    :param kernel: The kernel instance to convolve with.
    :type kernel: :class:`Kernel`
    :param sampling_rate: The sampling rate which will be used to discretize the
        kernel. It should be equal to the sampling rate used to obtain `binned`.
        The unit will typically be a frequency unit.
    :type sampling_rate: Quantity scalar
    :param mode:
        * 'same': The default which returns an array of the same size as
          `binned`
        * 'full': Returns an array with a bin for each shift where `binned` and
          the discretized kernel overlap by at least one bin.
        * 'valid': Returns only the discretization bins where the discretized
          kernel and `binned` completely overlap.
        See also `numpy.convolve
        <http://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html>`_.
    :type mode: {'same', 'full', 'valid'}
    :param dict kernel_discretization_params: Additional discretization
        arguments which will be passed to :func:`.discretize_kernel`.
    :returns: The smoothed representation of `binned`.
    :rtype: Quantity 1D
    """
    k = discretize_kernel(
        kernel, sampling_rate=sampling_rate, **kernel_discretization_params)
    return scipy.signal.convolve(binned, k, mode)


def st_convolve(
        train, kernel, sampling_rate, mode='same', binning_params=None,
        kernel_discretization_params=None):
    """ Convolves a :class:`neo.core.SpikeTrain` with a kernel.
    :param train: Spike train to convolve.
    :type train: :class:`neo.core.SpikeTrain`
    :param kernel: The kernel instance to convolve with.
    :type kernel: :class:`Kernel`
    :param sampling_rate: The sampling rate which will be used to bin
        the spike train. The unit will typically be a frequency unit.
    :type sampling_rate: Quantity scalar
    :param mode:
        * 'same': The default which returns an array covering the whole
          duration of the spike train `train`.
        * 'full': Returns an array with additional discretization bins in the
          beginning and end so that for each spike the whole discretized
          kernel is included.
        * 'valid': Returns only the discretization bins where the discretized
          kernel and spike train completely overlap.
        See also :func:`scipy.signal.convolve`.
    :type mode: {'same', 'full', 'valid'}
    :param dict binning_params: Additional discretization arguments which will
        be passed to :func:`.tools.bin_spike_trains`.
    :param dict kernel_discretization_params: Additional discretization
        arguments which will be passed to :func:`.discretize_kernel`.
    :returns: The convolved spike train, the boundaries of the discretization
        bins
    :rtype: (Quantity 1D, Quantity 1D with the inverse units of `sampling_rate`)
    """
    if binning_params is None:
        binning_params = {}
    if kernel_discretization_params is None:
        kernel_discretization_params = {}

    binned, bins = tools.bin_spike_trains(
        {0: [train]}, sampling_rate, **binning_params)
    binned = binned[0][0]
    #sampling_rate = binned.size / (bins[-1] - bins[0])
    result = smooth(
        binned, kernel, sampling_rate, mode, **kernel_discretization_params)

    assert (result.size - binned.size) % 2 == 0
    num_additional_bins = (result.size - binned.size) // 2

    if len(binned):
        bins = sp.linspace(
            bins[0] - num_additional_bins / sampling_rate,
            bins[-1] + num_additional_bins / sampling_rate,
            result.size + 1)
    else:
        bins = [] * pq.s

    return result, bins
