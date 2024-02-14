from PIL import ImageStat
import scipy

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

def get_stats(im):
    intensities = list(im.getdata())

    stats = ImageStat.Stat(im)

    median = stats.median[0]
    variance = stats.var[0]
    kurtosis = scipy.stats.kurtosis(intensities)
    skewness = scipy.stats.skew(intensities)
    
    return {
        "median": median,
        "variance": variance,
        "kurtosis": kurtosis,
        "skewness": skewness
    }