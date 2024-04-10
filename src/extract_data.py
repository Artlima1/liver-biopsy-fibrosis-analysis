from auxiliar_functions import get_color_components, get_stats, extract_central_half
from PIL import Image
import pandas as pd
import os

DATA_FOLDER = "../data"
IMAGES_FOLDER = DATA_FOLDER + '/split_images'

directory = os.fsencode(IMAGES_FOLDER)

metrics_df = pd.DataFrame(columns=[
    'filename',
    'healthy',
    'color_component',
    'median',
    'variance',
    'kurtosis',
    'skewness',
    'freq_radius'
])

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    
    healthy = "Saudavel" in filename

    im = Image.open(IMAGES_FOLDER + '/' + filename)

    im_cropped = extract_central_half(im)

    components = get_color_components(im_cropped)

    for component in components:
        img_stats = get_stats(components[component])
        
        metrics_df.loc[len(metrics_df)] = [
            filename,
            healthy,
            component,
            img_stats['median'],
            img_stats['variance'],
            img_stats['kurtosis'],
            img_stats['skewness'],
            img_stats['freq_radius']
        ]

metrics_df.to_csv(DATA_FOLDER + '/metrics.csv', index=False)