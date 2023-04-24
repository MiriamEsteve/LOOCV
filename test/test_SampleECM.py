import random

import pandas as pd
import numpy as np
import superdea as sDEA
import time

sd = random.randint(0, 1000) #int(time.time())
print(sd)

def append_df_to_excel(df, excel_path):
    try:
        df_excel = pd.read_excel(excel_path)
        result = pd.concat([df_excel, df], ignore_index=True)
        result.to_excel(excel_path, index=False)
    except:
        df.to_excel(excel_path, index=False)


exp = 10
for i in range(exp):
    sd = random.randint(0, 1000)  # int(time.time())
    print(sd)

    all_matrix = sDEA.sampleECM(sd).matrix_complete

    # Resultados de las (9)
    for matrix in all_matrix[400:]:
        if matrix["esc"][0] in [2, 3, 4, 5]: # (9)
            model = "CCR"
        else:
            model = "BCC"

        data_result = pd.DataFrame(
            columns=["exp", "esc", "N", "Psi", "Psi_value", "gamma", "rechazo_z_ECM", "acierto_x_ECM",
                     "rechazo_z_Ruggiero", "acierto_x_Ruggiero", "rechazo_z_superDEA", "acierto_x_superDEA",
                     "Sigificatives_ECM", "Sigificatives_Ruggiero", "Sigificatives_superDEA"])

        # Variables
        VAR = []
        y = ["y"]
        x = matrix.columns[matrix.columns.str.startswith('x')].values.tolist()
        x_without_z = x[:-1]

        # Result Data Frame
        data_result = data_result.append({"esc": matrix["esc"][0], "N": matrix["N"][0], "Psi": matrix["Psi"][0], "Psi_value": matrix["Psi_value"][0], "gamma": matrix["gamma"][0], "Sigificatives_ECM": None, "Sigificatives_Ruggiero": None, "Sigificatives_superDEA": None}, ignore_index=True)

        ####################################### ECM ##############################################3
        # #### TABLA 1
        start = time.time()
        model_ECM = sDEA.ECM(matrix.loc[:, x + y], VAR, y, model=model)
        result_ECM = model_ECM.fit_ECM_Backward()
        #result_ECM = model_ECM.fit_ECM_Forward()
        time_ECM1 = time.time() - start

        # Contar #rechazos de z (cuántas veces incluye a z = x4 como relevante)
        rechazo_z_ECM = 1 if x[-1] not in model_ECM.list_nosig else 0

        # #### TABLA 2
        start = time.time()
        model_ECM = sDEA.ECM(matrix.loc[:, x_without_z + y], VAR, y, model=model)
        result_ECM = model_ECM.fit_ECM_Backward()
        # result_ECM = model_ECM.fit_ECM_Forward()
        time_ECM2 = time.time() - start

        lst_sig = list(set(x_without_z) - set(model_ECM.list_nosig))

        # Contar #aciertos de x (cuántas veces incluye a x's como relevante)
        acierto_x_ECM = 1 if lst_sig == x_without_z else 0

        data_result.loc[len(data_result)-1, "rechazo_z_ECM"] = rechazo_z_ECM
        data_result.loc[len(data_result)-1,"acierto_x_ECM"] = acierto_x_ECM
        data_result.iat[len(data_result)-1, data_result.columns.get_loc("Sigificatives_ECM")] = lst_sig if len(lst_sig) !=0 else None



        ############################# Ruggiero ##########################################
        VAR = [random.choice(tuple(x))]

        # #### TABLA 1
        start = time.time()
        model_Ruggiero = sDEA.Ruggiero(matrix[x+y], VAR, y, model)
        dataRuggiero = model_Ruggiero.fit()
        lst_sig_Ruggiero = model_Ruggiero.VARCol
        time_Ruggiero1 = time.time() - start

        # Contar #rechazos de z (cuántas veces incluye a z = x4 como relevante)
        rechazo_z_Ruggiero = 1 if x[-1] in lst_sig_Ruggiero else 0

        # #### TABLA 2
        VAR = ["x1"]
        start = time.time()
        model_Ruggiero = sDEA.Ruggiero(matrix[x_without_z + y], VAR, y, model)
        dataRuggiero = model_Ruggiero.fit()
        lst_sig_Ruggiero = model_Ruggiero.VARCol
        time_Ruggiero2 = time.time() - start

        # Contar #acierto de x (cuántas veces incluye a x's como relevante)
        acierto_x_Ruggiero = 1 if lst_sig_Ruggiero == x_without_z else 0

        data_result.loc[len(data_result)-1,"rechazo_z_Ruggiero"] = rechazo_z_Ruggiero
        data_result.loc[len(data_result)-1,"acierto_x_Ruggiero"] = acierto_x_Ruggiero
        data_result.iat[len(data_result)-1,data_result.columns.get_loc("Sigificatives_Ruggiero")] = lst_sig_Ruggiero if len(lst_sig_Ruggiero) !=0 else None

        ############################# superDEA ##########################################
        # #### TABLA 1
        start = time.time()
        model_superDEA = sDEA.DEA(matrix[x+y], x, y)
        result, lst_sig = model_superDEA.superDEA()
        time_superDEA1 = time.time() - start

        # Contar #rechazos de z (cuántas veces incluye a z = x4 como relevante)
        rechazo_z_superDEA = 1 if x[-1] in lst_sig else 0

        # #### TABLA 2 - Sin z
        start = time.time()
        model_superDEA = sDEA.DEA(matrix[x_without_z + y], x_without_z, y)
        result, lst_sig = model_superDEA.superDEA()
        time_superDEA2 = time.time() - start

        # Contar #rechazos de x (cuántas veces incluye a x's como relevante)
        acierto_x_superDEA = 1 if lst_sig == x_without_z else 0

        data_result.loc[len(data_result)-1,"rechazo_z_superDEA"] = rechazo_z_superDEA
        data_result.loc[len(data_result)-1,"acierto_x_superDEA"] = acierto_x_superDEA
        data_result.iat[len(data_result)-1, data_result.columns.get_loc("Sigificatives_superDEA")] = lst_sig if len(lst_sig) !=0 else None

        data_result.loc[len(data_result) - 1, "exp"] = i

        #### Percentage x's
        for xi in x_without_z:
            data_result.loc[len(data_result) - 1, "ECM_" + str(xi)] = 1 if data_result.loc[len(data_result) - 1, "Sigificatives_ECM"] != None and xi in data_result.loc[len(data_result) - 1, "Sigificatives_ECM"] else 0
            data_result.loc[len(data_result) - 1, "Ruggiero_" + str(xi)] = 1 if data_result.loc[len(data_result) - 1, "Sigificatives_Ruggiero"] != None and xi in data_result.loc[len(data_result) - 1, "Sigificatives_Ruggiero"] else 0
            data_result.loc[len(data_result) - 1, "SuperDEA_" + str(xi)] = 1 if data_result.loc[len(data_result) - 1, "Sigificatives_superDEA"] != None and xi in data_result.loc[len(data_result) - 1, "Sigificatives_superDEA"] else 0

        #TIME
        data_result["time_ECM1"] = time_ECM1
        data_result["time_ECM2"] = time_ECM2
        data_result["time_Ruggiero1"] = time_Ruggiero1
        data_result["time_Ruggiero2"] = time_Ruggiero2
        data_result["time_superDEA1"] = time_superDEA1
        data_result["time_superDEA2"] = time_superDEA2

        # Write file
        path = r"C:/Users/Miriam_Esteve/Documents/CIO/Investigaciones/2022/superefficiencyDEA/exp/" + str(matrix["esc"][0]) + "_" + str(matrix["N"][0]) + "_" + str(matrix["Psi"][0]) + "_" + str(matrix["Psi_value"][0]) +  "_" + str(matrix["gamma"][0]) + "_result.xlsx"
        with open(path, 'a+'):
            append_df_to_excel(data_result, path)



