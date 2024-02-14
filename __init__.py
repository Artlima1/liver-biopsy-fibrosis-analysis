from auxiliar_functions import get_color_components, get_stats
from PIL import Image
import pandas as pd
import os

directory = os.fsencode('data/images')

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
    
    healthy = "saudavel" in filename

    im = Image.open('data/images/' + filename)
    components = get_color_components(im)

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

metrics_df.to_csv('data/metrics.csv', index=False)