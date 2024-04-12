from PIL import ImageStat
import scipy
import numpy as np

def get_color_components(im):
    R, G, B = im.split()
    H, S, V = im.convert("HSV").split()
    C, M, Y, K = im.convert("CMYK").split()
    Y_prime, Cb, Cr = im.convert("YCbCr").split()
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
        "Y'": Y_prime,
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


def extract_central_half(img):    
    # Get dimensions of the image
    width, height = img.size
    
    # Calculate the bounding box for the central half
    left = width // 4
    top = height // 4
    right = 3 * width // 4
    bottom = 3 * height // 4
    
    # Crop the image using the bounding box
    cropped_img = img.crop((left, top, right, bottom))
    
    return cropped_img

def split_image(original_image):
    # Open the image
    original_width, original_height = original_image.size
    
    # Calculate the width and height of each sub-image
    sub_width = original_width // 3
    sub_height = original_height // 3
    
    sub_images = []
    
    # Loop through each row and column to extract sub-images
    for y in range(3):
        for x in range(3):
            left = x * sub_width
            upper = y * sub_height
            right = left + sub_width
            lower = upper + sub_height
            
            # Crop the sub-image
            sub_image = original_image.crop((left, upper, right, lower))
            sub_images.append(sub_image)
    
    return sub_images
