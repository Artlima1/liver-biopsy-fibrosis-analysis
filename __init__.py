from auxiliar_functions import get_color_components, get_stats
from PIL import Image
import pandas as pd

im = Image.open('data/images/example.jpg')
components = get_color_components(im)

metrics_df = pd.DataFrame(columns=[
    'filename',
    'tissue_type',
    'color_component',
    'median',
    'variance',
    'kurtosis',
    'skewness'
])

for component in components:
    img_stats = get_stats(components[component])
    
    metrics_df.loc[len(metrics_df)] = [
        "example.jpg",
        "healthy",
        component,
        img_stats['median'],
        img_stats['variance'],
        img_stats['kurtosis'],
        img_stats['skewness']
    ]

metrics_df.to_csv('data/metrics.csv', index=False)