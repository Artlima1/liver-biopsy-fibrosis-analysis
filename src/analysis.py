import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt

DATA_FOLDER = "../data"

components = ["R","G","B","H","S","V","C","M","Y", "Y'","Cb","Cr","L","a","b"]
metrics = ["median","variance","kurtosis","skewness","freq_radius"]

metrics_df = pd.read_csv('../data/metrics.csv')
analysis_df = pd.DataFrame(columns=[
    'color_component',
    'metric',
    'rs_value',
    'p_value'
])

for component in components:
    healthy_component = metrics_df[(metrics_df.healthy == True) & (metrics_df.color_component == component)]
    fibrosis_component = metrics_df[(metrics_df.healthy == False) & (metrics_df.color_component == component)]
    for metric in metrics:
        healthy = healthy_component[metric].to_numpy()
        fibrosis = fibrosis_component[metric].to_numpy()

        data = [healthy, fibrosis]
        fig, ax = plt.subplots()
        ax.boxplot(data)
        ax.set_xticklabels(["Saudavel", "Fibrose"])
        plt.title(component + " - " + metric)
        plt.savefig(DATA_FOLDER + "/boxplots/" + component + " - " + metric + ".png")
        plt.close()

        rs = scipy.stats.ranksums(healthy, fibrosis)
        analysis_df.loc[len(analysis_df)] = [
            component,
            metric,
            rs.statistic,
            rs.pvalue
        ]


analysis_df.to_csv(DATA_FOLDER + '/analysis.csv', index=False)
analysis_df.to_excel(DATA_FOLDER + '/Analysis.xlsx')


