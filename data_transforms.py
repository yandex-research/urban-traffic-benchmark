"""
Data transformations that are almost like in scikit-learn preprocessing
(https://scikit-learn.org/1.5/api/sklearn.preprocessing.html) but can also transform torch tensors.
"""

from abc import ABC, abstractmethod
import warnings
import numpy as np
from scipy import stats
import torch


def torch_interp(x, xp, fp, extrapolation='constant'):
    """
    A 1d interpolation function like numpy interp but for torch tensors and with support for tensors with more than 1
    dimension (interpolation is done along the last dimension).

    Taken from https://github.com/pytorch/pytorch/issues/50334#issuecomment-2304751532.

    One-dimensional linear interpolation between monotonically increasing sample points, with extrapolation beyond
    sample points.

    Returns the one-dimensional piecewise linear interpolant to a function with given discrete data points (xp, fp),
    evaluated at x.

    The extrapolation argument determines how the values outside the range of xp are handled. The oprions are:
        - 'constant' (default): Use the boundary value of `fp` for `x` values outside `xp`.
        - 'linear': Extrapolate linearly beyond range of xp values.
    """
    m = torch.diff(fp) / torch.diff(xp)  # slope
    b = fp[..., :-1] - m * xp[..., :-1]  # offset
    indices = torch.searchsorted(xp, x, right=False)

    if extrapolation == 'constant':
        # Pad m and b to get constant values outside of xp range
        m = torch.cat([torch.zeros_like(m)[..., :1], m, torch.zeros_like(m)[..., :1]], dim=-1)
        b = torch.cat([fp[..., :1], b, fp[..., -1:]], dim=-1)
    else:  # extrapolation == 'linear'
        indices = torch.clamp(indices - 1, 0, m.shape[-1] - 1)

    f = m.gather(-1, indices) * x + b.gather(-1, indices)

    return f


def _handle_zeros_in_scale(scale_array: np.ndarray) -> np.ndarray:
    """
    Handles zeros in the array which will be used as a denominator in data transforms
    """
    near_zero_value = 10 * np.finfo(scale_array.dtype).eps
    return np.where(scale_array > near_zero_value, scale_array, 1)


class BaseDataTransform(ABC):
    """
    Data transforms subclassing this abstract base class can transform both numpy arrays and torch tensors. They can
    be fitted on numpy arrays (methods "fit" and "fit_transform" can be called only on a numpy array), but then can be
    moved to torch and transform torch tensors. Methods "numpy" and "torch" move the transform to numpy or torch,
    respectively. Method "to" moves torch transform to a specific device.
    """
    @abstractmethod
    def fit(self, x):
        """x should be a numpy array of shape [num_samples, num_features]."""
        pass

    @abstractmethod
    def transform(self, x):
        """x should be a numpy array or a torch tensor of shape [num_samples, num_features]."""
        pass

    def fit_transform(self, x):
        """x should be a numpy array of shape [num_samples, num_features]."""
        self.fit(x)

        return self.transform(x)

    @abstractmethod
    def inverse_transform(self, x):
        """x should be a numpy array or a torch tensor of shape [num_samples, num_features]."""
        pass

    @abstractmethod
    def torch(self):
        """
        After calling this method, the transform can be applied to torch tensors, but can no longer be applied to
        numpy arrays until "numpy" method is called.
        """
        pass

    @abstractmethod
    def numpy(self):
        """
        After calling this method, the transform can be applied to numpy arrays, but can no longer be applied to
        torch tensors until "torch" method is called.
        """
        pass

    @abstractmethod
    def to(self, device):
        """
        Moves all tensors necessary for applying data transform to device. Only works on a transform that has been
        moved to torch with "torch" method.
        """
        pass


class IdentityTransform(BaseDataTransform):
    def fit(self, x):
        pass

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x

    def torch(self):
        return self

    def numpy(self):
        return self

    def to(self, device):
        return self


class StandardScaler(BaseDataTransform):
    def fit(self, x):
        self.mean = np.nanmean(x, axis=0)
        self.mean = np.where(np.isnan(self.mean), 0, self.mean)
        self.std = _handle_zeros_in_scale(np.nanstd(x, axis=0, ddof=0))

    def transform(self, x):
        x -= self.mean
        x /= self.std

        return x

    def inverse_transform(self, x):
        x *= self.std
        x += self.mean

        return x

    def torch(self):
        self.mean = torch.from_numpy(self.mean)
        self.std = torch.from_numpy(self.std)

        return self

    def numpy(self):
        self.mean = self.mean.cpu().numpy()
        self.std = self.std.cpu().numpy()

        return self

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

        return self


class MinMaxScaler(BaseDataTransform):
    def fit(self, x):
        self.min = np.nanmin(x, axis=0)
        self.range = _handle_zeros_in_scale(np.nanmax(x, axis=0) - self.min)

    def transform(self, x):
        x -= self.min
        x /= self.range

        return x

    def inverse_transform(self, x):
        x *= self.range
        x += self.min

        return x

    def torch(self):
        self.min = torch.from_numpy(self.min)
        self.range = torch.from_numpy(self.range)

        return self

    def numpy(self):
        self.min = self.min.cpu().numpy()
        self.range = self.range.cpu().numpy()

        return self

    def to(self, device):
        self.min = self.min.to(device)
        self.range = self.range.to(device)

        return self


class RobustScaler(BaseDataTransform):
    def fit(self, x):
        first_quartile, median, third_quartile = np.nanquantile(x, q=(0.25, 0.5, 0.75), axis=0, method='linear')
        self.median = median
        self.iqr = _handle_zeros_in_scale(third_quartile - first_quartile)

    def transform(self, x):
        x -= self.median
        x /= self.iqr

        return x

    def inverse_transform(self, x):
        x *= self.iqr
        x += self.median

        return x

    def torch(self):
        self.median = torch.from_numpy(self.median)
        self.iqr = torch.from_numpy(self.iqr)

        return self

    def numpy(self):
        self.median = self.median.cpu().numpy()
        self.iqr = self.iqr.cpu().numpy()

        return self

    def to(self, device):
        self.median = self.median.to(device)
        self.iqr = self.iqr.to(device)

        return self


class QuantileTransform(BaseDataTransform):
    def __init__(self, distribution, num_quantiles=1001, eps=1e-7):
        if distribution not in ('normal', 'uniform'):
            raise ValueError('distribution argument should be either "normal" or "uniform".')

        self.distribution = distribution
        self.num_quantiles = num_quantiles
        self.eps = eps

        if distribution == 'normal':
            self.normal_dist = torch.distributions.Normal(loc=0, scale=1)
            self.clip_min = stats.norm.ppf(eps - np.spacing(1))
            self.clip_max = stats.norm.ppf(1 - (eps - np.spacing(1)))

    def fit(self, x):
        if self.num_quantiles > x.shape[0]:
            self.num_quantiles = x.shape[0]
            warnings.warn(f'The value of num_quantiles argument is {self.num_quantiles}, which is greater than the '
                          f'number of samples {x.shape[0]}. The number of quantiles used will be set to {x.shape[0]}.')

        probs = np.linspace(start=0, stop=1, num=self.num_quantiles, endpoint=True)
        quantiles = np.nanquantile(x, q=probs, axis=0)
        # Due to floating-point precision errors, quantiles can sometimes be not monotonically increasing:
        # https://github.com/numpy/numpy/issues/14685. In case this happens, the next line fixes it.
        quantiles = np.maximum.accumulate(quantiles)

        self.probs = probs
        self.quantiles = quantiles

    def transform(self, x):
        for i in range(x.shape[1]):
            x[:, i] = self._transform_col(x[:, i], quantiles=self.quantiles[:, i], inverse=False)

        return x

    def inverse_transform(self, x):
        for i in range(x.shape[1]):
            x[:, i] = self._transform_col(x[:, i], quantiles=self.quantiles[:, i], inverse=True)

        return x

    def _transform_col(self, col, quantiles, inverse):
        if not inverse:
            lower_bound_x = quantiles[0]
            upper_bound_x = quantiles[-1]
            lower_bound_y = 0
            upper_bound_y = 1
        else:
            lower_bound_x = 0
            upper_bound_x = 1
            lower_bound_y = quantiles[0]
            upper_bound_y = quantiles[-1]

        if inverse and self.distribution == 'normal':
            if isinstance(col, torch.Tensor):
                col = self.normal_dist.cdf(col)
            else:
                col = stats.norm.cdf(col)

        if self.distribution == 'normal':
            lower_bounds_mask = (col - self.eps < lower_bound_x)
            upper_bounds_mask = (col + self.eps > upper_bound_x)
        else:
            lower_bounds_mask = (col == lower_bound_x)
            upper_bounds_mask = (col == upper_bound_x)

        if not inverse:
            if isinstance(col, torch.Tensor):
                col = 0.5 * (
                        torch_interp(xp=quantiles, fp=self.probs, x=col) -
                        torch_interp(xp=-quantiles.flip(0), fp=-self.probs.flip(0), x=-col)
                )
            else:
                col = 0.5 * (
                        np.interp(xp=quantiles, fp=self.probs, x=col) -
                        np.interp(xp=-quantiles[::-1], fp=-self.probs[::-1], x=-col)
                )
        else:
            if isinstance(col, torch.Tensor):
                col = torch_interp(xp=self.probs, fp=quantiles, x=col)
            else:
                col = np.interp(xp=self.probs, fp=quantiles, x=col)

        col[upper_bounds_mask] = upper_bound_y
        col[lower_bounds_mask] = lower_bound_y

        if not inverse and self.distribution == 'normal':
            if isinstance(col, torch.Tensor):
                col = self.normal_dist.icdf(col).clip(min=self.clip_min, max=self.clip_max)
            else:
                col = stats.norm.ppf(col).clip(min=self.clip_min, max=self.clip_max)

        return col

    def torch(self):
        self.probs = torch.from_numpy(self.probs)
        self.quantiles = torch.from_numpy(self.quantiles)

        return self

    def numpy(self):
        self.probs = self.probs.cpu().numpy()
        self.quantiles = self.quantiles.cpu().numpy()

        return self

    def to(self, device):
        self.probs = self.probs.to(device)
        self.quantiles = self.quantiles.to(device)

        return self
