import random

import pandas as pd
import numpy as np
import superdea as sDEA
import time

#Years
years = [i for i in range(2001, 2008)] #1995-2007

#Read dataset
path="C:/Users/Miriam_Esteve/OneDrive - Fundación Universitaria San Pablo CEU/Documents/CIO/Investigaciones/2022/superefficiencyDEA/paquete/dataset/empirical_data.xlsx"
for year in years:
    data = pd.read_excel(path, sheet_name=str(year))
    x = data.columns[data.columns.str.startswith('x')].values.tolist()
    y = ["y"]

    #CVDEA
    model_superDEA = sDEA.DEA(data[x + y], x, y)
    result, lst_sig = model_superDEA.superDEA()

    print("----------------------------------------------> Year: ", year, " - Best model: ", lst_sig)



