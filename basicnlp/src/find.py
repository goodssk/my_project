# encoding: utf-8
import numpy as np
import pandas as pd
import re
from numpy import log, min

df = pd.read_csv('result2.csv')
df = df[~df.word.duplicated(keep='last')]
print(df.to_excel('result.xlsx'))
