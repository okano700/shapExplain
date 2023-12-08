import os
import pandas as pd 

torun = pd.read_csv('~/shapExplain/torun_DA_YAHOO.csv')

prestr =  '/lustre/eyokano/datasets/Yahoo/ydata-labeled-time-series-anomalies-v1_0/'

for ind, ds in torun.iterrows():
    for i in range(5):
        os.system(f'python ~/MetaLearningModels/runDeepAnt.py  --path {prestr+ds["Dataset"]} --WL {ds["WL"]} --n {ds["n"]} --i {i} --source YAHOO >> log/log_YAHOODeepAnt.log')
