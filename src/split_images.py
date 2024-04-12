import os
from PIL import Image
from auxiliar_functions import split_image, save_sub_images

DATA_FOLDER = "../data"
INPUT_IMAGES_FOLDER = DATA_FOLDER + '/images'
OUTPUT_IMAGES_FOLDER = DATA_FOLDER + '/split_images'

directory = os.fsencode(INPUT_IMAGES_FOLDER)

for file in os.listdir(directory):

    filename = os.fsdecode(file)

    original_img = Image.open(INPUT_IMAGES_FOLDER + '/' + filename)

    # Split the image into sub-images
    sub_images = split_image(original_img)
    
    # Save the sub-images
    for i, sub_image in enumerate(sub_images):
        sub_image.save(f"{OUTPUT_IMAGES_FOLDER}/{filename[:-4]}_{i}.tiff", format="TIFF")