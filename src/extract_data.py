import auxiliar_functions as aux
from PIL import Image
import pandas as pd
import os

DATA_FOLDER = "../data"
IMAGES_FOLDER = DATA_FOLDER + '/images'

directory = os.fsencode(IMAGES_FOLDER)
data = []

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)

    healthy = "Saudavel" in filename

    im = Image.open(IMAGES_FOLDER + '/' + filename)

    components = aux.get_color_components(im)

    i = len(data)
    data.insert(i, {"filename": filename, "healthy": healthy})
    for component in components:
        img_stats = aux.get_stats(components[component], component)
        data[i].update(img_stats)
        
metrics_df = pd.DataFrame(data)
metrics_df.to_csv(DATA_FOLDER + '/metrics.csv', index=False)