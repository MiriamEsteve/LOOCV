import random

import pandas as pd
import numpy as np
import superdea as sDEA
import time

sd = random.randint(0, 1000)  # int(time.time())
print(sd)

all_matrix = sDEA.sampleECM(sd).matrix_complete
vuelta = 0
for matrix in all_matrix:
    if matrix["Psi"][0] != "half_normal":
        continue

    # Write file
    path = "C:/Users/Miriam_Esteve/Documents/CIO/Investigaciones/2022/superefficiencyDEA/exp-con_margenes/" + str(
    matrix["esc"][0]) + "_" + str(matrix["N"][0]) + "_" + str(matrix["Psi"][0]) + "_" + str(
    matrix["Psi_value"][0]) + "_" + str(matrix["gamma"][0]) + "_result.xlsx"
    matrix = pd.read_excel(path)

    #Add experiments
    while len(matrix) < 100:
        matrix = matrix.append(matrix.iloc[:, :]).reset_index(drop=True)
        matrix.to_excel(path)

    #Calculate %
    vuelta +=1
    if vuelta == 1:
        # Result Data Frame
        data_result = pd.DataFrame([matrix.iloc[:, 7:].mean(axis = 0)], columns = matrix.columns)
        data_result.iloc[0, :7] = matrix.iloc[0, :7]
    else:
        # Result Data Frame
        data_result = data_result.append(matrix.iloc[:, 7:].mean(axis = 0), ignore_index=True)
        data_result.iloc[vuelta - 1, :7] = matrix.iloc[0, :7]

data_result.to_excel("C:/Users/Miriam_Esteve/Documents/CIO/Investigaciones/2022/superefficiencyDEA/exp-con_margenes/results_total_half-normal.xlsx")







