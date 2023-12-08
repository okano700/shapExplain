import glob
import os
from utils.TSds import TSds
from utils.find_frequency import get_period

#ds = glob.glob("~/datsets/UCR_Anomaly_FullData/*.txt")
datasets = glob.glob('/lustre/eyokano/datasets/Yahoo/ydata-labeled-time-series-anomalies-v1_0/A3Benchmark/*.csv')
print('sadsad')
print(datasets, flush=True)
#print('aqui')
datasets.sort()


for p in datasets:
    ds = TSds.read_YAHOO(p)
    for f in get_period(ds.ts[:ds.train_split],3):
        for n in [2,3,5]:
            for i in range(5):
                #os.system(f'python t.py --path {i} --ds UCR --WL 100 --n 5 --i 1')
                os.system(f'python ~/MetaLearningModels/runDeepAnt.py  --path {p} --WL {f} --n {n} --i {i} --source YAHOO >> log/log_YAHOODeepAnt.log')
                #print(f, n, i, p)
