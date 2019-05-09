# -*- coding: utf-8 -*-
"""
Created on Thu May  9 09:36:52 2019

@author: PIANDT
"""

#1-feature engineering, choosing classifier, 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

fruitsData = pd.read_table('fruit_data_with_colors.txt')

#summary of first 5 rows
fruitsData.head()

#summarized analyses of quantitative data
fruitsData.describe()

# create a dictionary of fruit type (fruit_label) which is numeric
# and map to the actual fruit name (fruit_name). 
#This gives a true representation of our output labels or anything we want to classify
individualFruitNames = dict(zip(fruitsData.fruit_label.unique(), fruitsData.fruit_name.unique()))   

#The fruitsData has information about fruit name (numeric and string), height, 
# mass, color score and width of selected fruits.
#heights -- how tall is the fruit 
#widths -- how wide is the fruit
#mass -- how heavy is the fruit