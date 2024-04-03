from PIL import ImageStat
import scipy
import numpy as np

def get_color_components(im):
    R, G, B = im.split()
    H, S, V = im.convert("HSV").split()
    C, M, Y, K = im.convert("CMYK").split()
    Y, Cb, Cr = im.convert("YCbCr").split()
    L, a, b = im.convert("LAB").split()

    components = {
        "R": R,
        "G": G,
        "B": B,
        "H": H,
        "S": S,
        "V": V,
        "C": C,
        "M": M,
        "Y": Y,
        "Cb": Cb,
        "Cr": Cr,
        "L": L,
        "a": a,
        "b": b
    }

    return components

def sum_values_inside_circle(array, radius):
    center = (np.array(array.shape) - 1) / 2  # Center of the array
    y, x = np.ogrid[:array.shape[0], :array.shape[1]]
    mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
    return np.sum(array[mask])

def get_radius(image):
    img_np = np.asarray(image)
    ft = scipy.fft.fft2(img_np)
    ft = scipy.fft.fftshift(ft)
    mag = np.abs(ft)

    half_tot = np.sum(mag)/2
    sum = 0
    rad = 0

    while sum < half_tot:
        rad += 1
        sum = sum_values_inside_circle(mag, rad)

    return rad

def get_stats(im):
    intensities = list(im.getdata())

    stats = ImageStat.Stat(im)

    median = stats.median[0]
    variance = stats.var[0]
    kurtosis = scipy.stats.kurtosis(intensities)
    skewness = scipy.stats.skew(intensities)
    rad = get_radius(im)
    
    return {
        "median": median,
        "variance": variance,
        "kurtosis": kurtosis,
        "skewness": skewness,
        "freq_radius": rad
    }


