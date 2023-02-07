from typing import Union
import audioread
import os
import numpy as np

from typing import Tuple
from typing import Any
from typing import TypeVar
import soxr
import soundfile as sf
import scipy
import scipy.signal
import scipy.io
import scipy.io.wavfile


from numpy.lib.stride_tricks import as_strided

# Constrain STFT block sizes to 256 KB
MAX_MEM_BLOCK = 2 ** 8 * 2 ** 10

class Deprecated(object):
    """A dummy class to catch usage of deprecated variable names"""

    def __repr__(self) -> str:
        return "<DEPRECATED parameter>"

def valid_audio(y: np.ndarray, *, mono: Union[bool, Deprecated] = Deprecated()) -> bool:  

    return True



def dtype_r2c(d, *, default=np.complex64):
   
    mapping = {
        np.dtype(np.float32): np.complex64,
        np.dtype(np.float64): np.complex128,
        np.dtype(float): np.dtype(complex).type,
    }

    # If we're given a complex type already, return it
    dt = np.dtype(d)
    if dt.kind == "c":
        return dt

    # Otherwise, try to map the dtype.
    # If no match is found, return the default.
    return np.dtype(mapping.get(dt, default))


def buf_to_float(
    x: np.ndarray, *, n_bytes: int = 2, dtype = np.float32
) -> np.ndarray:
   
    # Invert the scale of the data
    scale = 1.0 / float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = "<i{:d}".format(n_bytes)

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)


def phase_vocoder(D, *, rate, hop_length=None, n_fft=None):
    

    if n_fft is None:
        n_fft = 2 * (D.shape[-2] - 1)

    if hop_length is None:
        hop_length = int(n_fft // 4)

    time_steps = np.arange(0, D.shape[-1], rate, dtype=np.float64)

    # Create an empty output array
    shape = list(D.shape)
    shape[-1] = len(time_steps)
    d_stretch = np.zeros_like(D, shape=shape)

    # Expected phase advance in each bin
    phi_advance = np.linspace(0, np.pi * hop_length, D.shape[-2])

    # Phase accumulator; initialize to the first sample
    phase_acc = np.angle(D[..., 0])

    # Pad 0 columns to simplify boundary logic
    padding = [(0, 0) for _ in D.shape]
    padding[-1] = (0, 2)
    D = np.pad(D, padding, mode="constant")

    for (t, step) in enumerate(time_steps):

        columns = D[..., int(step) : int(step + 2)]

        # Weighting for linear magnitude interpolation
        alpha = np.mod(step, 1.0)
        mag = (1.0 - alpha) * np.abs(columns[..., 0]) + alpha * np.abs(columns[..., 1])

        # Store to output array
        d_stretch[..., t] = mag * np.exp(1.0j * phase_acc)

        # Compute phase advance
        dphase = np.angle(columns[..., 1]) - np.angle(columns[..., 0]) - phi_advance

        # Wrap to -pi:pi range
        dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))

        # Accumulate phase
        phase_acc += phi_advance + dphase

    return d_stretch


def audioread_load(path, offset, duration, dtype):
    """Load an audio buffer using audioread.

    This loads one block at a time, and then concatenates the results.
    """

    buf = []

    if isinstance(path, tuple(audioread.available_backends())):
        # If we have an audioread object already, don't bother opening
        reader = path
    else:
        # If the input was not an audioread object, try to open it
        reader = audioread.audio_open(path)

    with reader as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels

        s_start = int(np.round(sr_native * offset)) * n_channels

        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + (int(np.round(sr_native * duration)) * n_channels)

        n = 0

        for frame in input_file:
            frame = buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)

            if n < s_start:
                # offset is after the current frame
                # keep reading
                continue

            if s_end < n_prev:
                # we're off the end.  stop reading
                break

            if s_end < n:
                # the end is in this frame.  crop.
                frame = frame[: int(s_end - n_prev)]  # pragma: no cover

            if n_prev <= s_start <= n:
                # beginning is in this frame
                frame = frame[(s_start - n_prev) :]

            # tack on the current frame
            buf.append(frame)

    if buf:
        y = np.concatenate(buf)
        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
    else:
        y = np.empty(0, dtype=dtype)

    return y, sr_native

def soundfile_load(path, offset, duration, dtype):
    """Load an audio buffer using soundfile."""
    if isinstance(path, sf.SoundFile):
        # If the user passed an existing soundfile object,
        # we can use it directly
        context = path
    else:
        # Otherwise, create the soundfile object
        context = sf.SoundFile(path)

    with context as sf_desc:
        sr_native = sf_desc.samplerate
        if offset:
            # Seek to the start of the target read
            sf_desc.seek(int(offset * sr_native))
        if duration is not None:
            frame_duration = int(duration * sr_native)
        else:
            frame_duration = -1

        # Load the target number of frames, and transpose to match librosa form
        y = sf_desc.read(frames=frame_duration, dtype=dtype, always_2d=False).T

    return y, sr_native

def frame(x, *, frame_length, hop_length, axis=-1, writeable=False, subok=False):
    

    x = np.array(x, copy=False, subok=subok)

    if x.shape[axis] < frame_length:
        raise ParameterError(
            "Input is too short (n={:d})"
            " for frame_length={:d}".format(x.shape[axis], frame_length)
        )

    if hop_length < 1:
        raise ParameterError("Invalid hop_length: {:d}".format(hop_length))

    # put our new within-frame axis at the end for now
    out_strides = x.strides + tuple([x.strides[axis]])

    # Reduce the shape on the framing axis
    x_shape_trimmed = list(x.shape)
    x_shape_trimmed[axis] -= frame_length - 1

    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = as_strided(
        x, strides=out_strides, shape=out_shape, subok=subok, writeable=writeable
    )

    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1

    xw = np.moveaxis(xw, -1, target_axis)

    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    return xw[tuple(slices)]

def to_mono(y: np.ndarray) -> np.ndarray:
 
    # Validate the buffer.  Stereo is ok here.
    valid_audio(y, mono=False)
    print(type(y))

    if y.ndim > 1:
        y = np.mean(y, axis=tuple(range(y.ndim - 1)))
        print((y))
    
    return y
def fix_length(
    data: np.ndarray, *, size: int, axis: int = -1, **kwargs: Any
) -> np.ndarray:
    

    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    if n > size:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, size)
        return data[tuple(slices)]

    elif n < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - n)
        return np.pad(data, lengths, **kwargs)

    return data

def expand_to(x, *, ndim, axes):
   
    # Force axes into a tuple

    try:
        axes = tuple(axes)
    except TypeError:
        axes = tuple([axes])

    if len(axes) != x.ndim:
        raise ParameterError(
            "Shape mismatch between axes={} and input x.shape={}".format(axes, x.shape)
        )

    if ndim < x.ndim:
        raise ParameterError(
            "Cannot expand x.shape={} to fewer dimensions ndim={}".format(x.shape, ndim)
        )

    shape = [1] * ndim
    for i, axi in enumerate(axes):
        shape[axi] = x.shape[i]

    return x.reshape(shape)

def resample(
    y: np.ndarray,    
    orig_sr: float,
    target_sr: float,
    res_type: str = "soxr_hq",
    fix: bool = True,
    scale: bool = False,
    axis: int = -1,
    **kwargs: Any,
) -> np.ndarray:
    

    # First, validate the audio buffer
    valid_audio(y, mono=False)

    if orig_sr == target_sr:
        return y

    ratio = float(target_sr) / float(orig_sr)
    
    print(type(y))

    n_samples = int(np.ceil(y.shape[axis] * ratio))

    if res_type in ("scipy", "fft"):
        y_hat = scipy.signal.resample(y, n_samples, axis=axis)
    elif res_type == "polyphase":
        if int(orig_sr) != orig_sr or int(target_sr) != target_sr:
            pass
            # raise ParameterError(
            #     "polyphase resampling is only supported for integer-valued sampling rates."
            # )

        # For polyphase resampling, we need up- and down-sampling ratios
        # We can get those from the greatest common divisor of the rates
        # as long as the rates are integrable
        orig_sr = int(orig_sr)
        target_sr = int(target_sr)
        gcd = np.gcd(orig_sr, target_sr)
        y_hat = scipy.signal.resample_poly(
            y, target_sr // gcd, orig_sr // gcd, axis=axis
        )
    elif res_type in (
        "linear",
        "zero_order_hold",
        "sinc_best",
        "sinc_fastest",
        "sinc_medium",
    ):
        # Use numpy to vectorize the resampler along the target axis
        # This is because samplerate does not support ndim>2 generally.
        y_hat = np.apply_along_axis(
            samplerate.resample, axis=axis, arr=y, ratio=ratio, converter_type=res_type
        )
    elif res_type.startswith("soxr"):
        # Use numpy to vectorize the resampler along the target axis
        # This is because soxr does not support ndim>2 generally.
        y_hat = np.apply_along_axis(
            soxr.resample,
            axis=axis,
            arr=y,
            in_rate=orig_sr,
            out_rate=target_sr,
            quality=res_type,
        )
    else:
        y_hat = resampy.resample(y, orig_sr, target_sr, filter=res_type, axis=axis)

    if fix:
        y_hat = fix_length(y_hat, size=n_samples, axis=axis, **kwargs)

    if scale:
        y_hat /= np.sqrt(ratio)

    # Match dtypes
    return np.asarray(y_hat, dtype=y.dtype)

def yakr_load(path) -> Tuple[np.ndarray, float]:
    
    sr:float = 22050
    mono: bool = True
    offset: float = 0.0
    duration = None
    dtype = np.float32
    res_type: str = "soxr_hq"

    dtype = np.float32
    if isinstance(path, tuple(audioread.available_backends())):
        print('1111111111111111111111111')
        # Force the audioread loader if we have a reader object already
        y, sr_native = audioread_load(path, offset, duration, dtype)
    else:
        print('2222222222222222222222')
        # Otherwise try soundfile first, and then fall back if necessary
        y, sr_native = soundfile_load(path, offset, duration, dtype)
        print(type(y))

    # Final cleanup for dtype and contiguity
    if mono:
        y = to_mono(y)
        print(type(y))


    if sr is not None:
       
        y = resample(y, float(sr_native), float(sr), res_type)

    else:
        sr = sr_native

    return y, sr

def __overlap_add(y, ytmp, hop_length):
   
    n_fft = ytmp.shape[-2]
    for frame in range(ytmp.shape[-1]):
        sample = frame * hop_length
        y[..., sample : (sample + n_fft)] += ytmp[..., frame]


def istft(
    stft_matrix,
    *,
    hop_length=None,
    win_length=None,
    n_fft=None,
    window="hann",
    center=True,
    dtype=None,
    length=None,
):
    

    if n_fft is None:
        n_fft = 2 * (stft_matrix.shape[-2] - 1)

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    ifft_window = get_window(window, win_length, fftbins=True)

    # Pad out to match n_fft, and add broadcasting axes
    ifft_window = pad_center(ifft_window, size=n_fft)
    ifft_window = expand_to(ifft_window, ndim=stft_matrix.ndim, axes=-2)

    # For efficiency, trim STFT frames according to signal length if available
    if length:
        if center:
            padded_length = length + int(n_fft)
        else:
            padded_length = length
        n_frames = min(stft_matrix.shape[-1], int(np.ceil(padded_length / hop_length)))
    else:
        n_frames = stft_matrix.shape[-1]

    if dtype is None:
        dtype = dtype_c2r(stft_matrix.dtype)

    shape = list(stft_matrix.shape[:-2])
    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    shape.append(expected_signal_len)
    y = np.zeros(shape, dtype=dtype)

    n_columns = MAX_MEM_BLOCK // (
        np.prod(stft_matrix.shape[:-1]) * stft_matrix.itemsize
    )
    n_columns = max(n_columns, 1)

    #fft = get_fftlib()
    fft = np.fft    

    frame = 0
    for bl_s in range(0, n_frames, n_columns):
        bl_t = min(bl_s + n_columns, n_frames)

        # invert the block and apply the window function
        ytmp = ifft_window * fft.irfft(stft_matrix[..., bl_s:bl_t], n=n_fft, axis=-2)

        # Overlap-add the istft block starting at the i'th frame
        __overlap_add(y[..., frame * hop_length :], ytmp, hop_length)

        frame += bl_t - bl_s

    # Normalize by sum of squared window
    ifft_window_sum = window_sumsquare(
        window=window,
        n_frames=n_frames,
        win_length=win_length,
        n_fft=n_fft,
        hop_length=hop_length,
        dtype=dtype,
    )

    approx_nonzero_indices = ifft_window_sum > tiny(ifft_window_sum)
    y[..., approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

    if length is None:
        # If we don't need to control length, just do the usual center trimming
        # to eliminate padded data
        if center:
            y = y[..., int(n_fft // 2) : -int(n_fft // 2)]
    else:
        if center:
            # If we're centering, crop off the first n_fft//2 samples
            # and then trim/pad to the target length.
            # We don't trim the end here, so that if the signal is zero-padded
            # to a longer duration, the decay is smooth by windowing
            start = int(n_fft // 2)
        else:
            # If we're not centering, start at 0 and trim/pad as necessary
            start = 0

        y = fix_length(y[..., start:], size=length)

    return y




def tiny(x):
   

    # Make sure we have an array view
    x = np.asarray(x)

    # Only floating types generate a tiny
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(
        x.dtype, np.complexfloating
    ):
        dtype = x.dtype
    else:
        dtype = np.float32

    return np.finfo(dtype).tiny


def normalize(S, *, norm=np.inf, axis=0, threshold=None, fill=None):
   
    # Avoid div-by-zero
    if threshold is None:
        threshold = tiny(S)

    elif threshold <= 0:
        raise ParameterError(
            "threshold={} must be strictly " "positive".format(threshold)
        )

    if fill not in [None, False, True]:
        raise ParameterError("fill={} must be None or boolean".format(fill))

    if not np.all(np.isfinite(S)):
        raise ParameterError("Input must be finite")

    # All norms only depend on magnitude, let's do that first
    mag = np.abs(S).astype(float)

    # For max/min norms, filling with 1 works
    fill_norm = 1

    if norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)

    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)

    elif norm == 0:
        if fill is True:
            raise ParameterError("Cannot normalize with norm=0 and fill=True")

        length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)

    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag ** norm, axis=axis, keepdims=True) ** (1.0 / norm)

        if axis is None:
            fill_norm = mag.size ** (-1.0 / norm)
        else:
            fill_norm = mag.shape[axis] ** (-1.0 / norm)

    elif norm is None:
        return S

    else:
        raise ParameterError("Unsupported norm: {}".format(repr(norm)))

    # indices where norm is below the threshold
    small_idx = length < threshold

    Snorm = np.empty_like(S)
    if fill is None:
        # Leave small indices un-normalized
        length[small_idx] = 1.0
        Snorm[:] = S / length

    elif fill:
        # If we have a non-zero fill value, we locate those entries by
        # doing a nan-divide.
        # If S was finite, then length is finite (except for small positions)
        length[small_idx] = np.nan
        Snorm[:] = S / length
        Snorm[np.isnan(Snorm)] = fill_norm
    else:
        # Set small values to zero by doing an inf-divide.
        # This is safe (by IEEE-754) as long as S is finite.
        length[small_idx] = np.inf
        Snorm[:] = S / length

    return Snorm

def __window_ss_fill(x, win_sq, n_frames, hop_length):  # pragma: no cover
    """Helper function for window sum-square calculation."""

    n = len(x)
    n_fft = len(win_sq)
    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]

def window_sumsquare(
    *,
    window,
    n_frames,
    hop_length=512,
    win_length=None,
    n_fft=2048,
    dtype=np.float32,
    norm=None,
):
   
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length)
    win_sq = normalize(win_sq, norm=norm) ** 2
    win_sq = pad_center(win_sq, size=n_fft)

    # Fill the envelope
    __window_ss_fill(x, win_sq, n_frames, hop_length)

    return x



def pad_center(data, *, size, axis=-1, **kwargs):
    

    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ParameterError(
            ("Target size ({:d}) must be " "at least input size ({:d})").format(size, n)
        )

    return np.pad(data, lengths, **kwargs)

def is_positive_int(x: float) -> bool:
    

    # Check type first to catch None values.
    return isinstance(x, (int, np.integer)) and (x > 0)

def get_window(window, Nx, *, fftbins=True):
    
    if callable(window):
        return window(Nx)

    elif isinstance(window, (str, tuple)) or np.isscalar(window):
        # TODO: if we add custom window functions in librosa, call them here

        return scipy.signal.get_window(window, Nx, fftbins=fftbins)

    elif isinstance(window, (np.ndarray, list)):
        if len(window) == Nx:
            return np.asarray(window)

        raise ParameterError(
            "Window size mismatch: " "{:d} != {:d}".format(len(window), Nx)
        )
    else:
        raise ParameterError("Invalid window specification: {}".format(window))


def stft_logic(
    y,
    *,
    n_fft=2048,
    hop_length=None,
    win_length=None,
    window="hann",
    center=True,
    dtype=None,
    pad_mode="constant"):
   
      # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    # Check audio is valid
    valid_audio(y, mono=False)

    fft_window = get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = pad_center(fft_window, size=n_fft)

    # Reshape so that the window can be broadcast
    fft_window = expand_to(fft_window, ndim=1 + y.ndim, axes=-2)

    # Pad the time series so that frames are centered
    if center:
        if n_fft > y.shape[-1]:
            warnings.warn(
                "n_fft={} is too small for input signal of length={}".format(
                    n_fft, y.shape[-1]
                ),
                stacklevel=2,
            )

        padding = [(0, 0) for _ in range(y.ndim)]
        padding[-1] = (int(n_fft // 2), int(n_fft // 2))
        y = np.pad(y, padding, mode=pad_mode)

    elif n_fft > y.shape[-1]:
        raise ParameterError(
            "n_fft={} is too large for input signal of length={}".format(
                n_fft, y.shape[-1]
            )
        )

    # Window the time series.
    y_frames = frame(y, frame_length=n_fft, hop_length=hop_length)

    # fft = get_fftlib()

    fft = np.fft

    if dtype is None:
        dtype = dtype_r2c(y.dtype)

    # Pre-allocate the STFT matrix
    shape = list(y_frames.shape)
    shape[-2] = 1 + n_fft // 2
    stft_matrix = np.empty(shape, dtype=dtype, order="F")

    n_columns = MAX_MEM_BLOCK // (
        np.prod(stft_matrix.shape[:-1]) * stft_matrix.itemsize
    )
    n_columns = max(n_columns, 1)

    for bl_s in range(0, stft_matrix.shape[-1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[-1])

        stft_matrix[..., bl_s:bl_t] = fft.rfft(
            fft_window * y_frames[..., bl_s:bl_t], axis=-2
        )
    return stft_matrix

def time_stretch(y: np.ndarray,rate: float, **kwargs: Any) -> np.ndarray:
      
    # Construct the short-term Fourier transform (STFT)
    stft = stft_logic(y, **kwargs)

    # # Stretch by phase vocoding
    stft_stretch = phase_vocoder(
        stft,
        rate=rate,
        hop_length=kwargs.get("hop_length", None),
        n_fft=kwargs.get("n_fft", None),
    )

    # # Predict the length of y_stretch
    len_stretch = int(round(y.shape[-1] / rate))

    # # Invert the STFT
    y_stretch = istft(stft_stretch, dtype=y.dtype, length=len_stretch, **kwargs)

    return y_stretch
