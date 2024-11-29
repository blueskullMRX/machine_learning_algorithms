from feature_selection import corr_filter, corr_filter_between
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = df = pd.read_csv('pd_speech_features.csv', header=1).drop('id',axis='columns')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)


if __name__ == '__main__': 
    pass