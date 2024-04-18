import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt
import auxiliar_functions as aux

DATA_FOLDER = "../data"

metrics_df = pd.read_csv('../data/metrics.csv')

metrics = metrics_df.columns[2:]

analysis = []

for metric in metrics:
    healthy = metrics_df[(metrics_df.healthy == True)][metric]
    fibrosis = metrics_df[(metrics_df.healthy == False)][metric]
    
    aux.plot_boxplot(healthy, fibrosis, metric, DATA_FOLDER)

    rs = scipy.stats.wilcoxon(healthy, fibrosis)
    analysis.insert(len(analysis),{
        "metric": metric,
        "statistic": rs.statistic,
        "p_value": rs.pvalue
    })

analysis_df = pd.DataFrame(analysis)

analysis_df.to_csv(DATA_FOLDER + '/analysis.csv', index=False)
analysis_df.to_excel(DATA_FOLDER + '/Analysis.xlsx')