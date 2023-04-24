import numpy as np
import pandas as pd
from math import e
from scipy.stats import truncnorm

class sampleECM:
    def __init__(self, seed):
        self.seed = seed

        #List of Data Frame with all simulations (450)
        self.matrix_complete = []

        # Gamma esc == 1
        self.gamma_list = [round(i, 3) for i in np.arange(0.1, 1.1, 0.1)]

        for N in [50, 100, 200]: #Tamaños muestrales
            for inef in ["half_normal", "exp", "gamma"]: #Inefficiencies
                if inef == "half_normal" or inef == "exp":
                    Psi_value = [0.8, 0.4]
                else:
                    Psi_value = [1.74, 3.47]
                for psi_value in Psi_value: # Valores de ineficiencia
                    # 10 funciones con 1 input. Distintos gammas
                    sesc = 1
                    for gamma in self.gamma_list:
                        print("##########  N = ", N, " - Psi = ", inef, " - Psi_value = ", psi_value, " - esc = ", sesc,
                              " - gamma = ", gamma)
                        self.matrix = sECM(self.seed, sesc, N, inef, psi_value, gamma).data
                        self.matrix["esc"] = sesc
                        self.matrix["N"] = N
                        self.matrix["Psi"] = inef
                        self.matrix["Psi_value"] = psi_value
                        self.matrix["gamma"] = gamma
                        self.matrix_complete.append(self.matrix)

                    # (9), (10), (11), (12)
                    for sesc in range(2, 17):
                        print("##########  N = ", N, " - Psi = ", inef, " - Psi_value = ", psi_value, " - esc = ", sesc)
                        self.matrix = sECM(self.seed, sesc, N, inef, psi_value).data
                        self.matrix["esc"] = sesc
                        self.matrix["N"] = N
                        self.matrix["Psi"] = inef
                        self.matrix["Psi_value"] = psi_value
                        self.matrix["gamma"] = None
                        self.matrix_complete.append(self.matrix)

class sECM:
    def __init__(self, seed, esc, N, Psi, value_Psi, gamma = None):
        self.seed = seed
        # Seed random
        np.random.seed(self.seed)

        self.esc = esc
        self.N = N
        self.type_inef = Psi
        self.value_inef = value_Psi

        # DataFrame vacio
        self.data = pd.DataFrame()

        # Generate nX. # Se añade un input no relevante z
        if self.esc == 1:
            self.nX = 1 + 1
        elif self.esc in range(1, 10):
            self.nX = 3 + 1
        else:
            self.nX = 5 + 1
        for x in range(self.nX):
            self.data["x" + str(x + 1)] = np.random.uniform(5, 15, self.N)

        # Generate inefficiencies
        if self.type_inef == "half_normal":
            self.u = abs(np.random.normal(self.value_inef, self.N))
        elif self.type_inef == "exp":
            self.u = np.random.exponential(self.value_inef, self.N)
        else: # gamma
            self.u = np.random.gamma(1.25, self.value_inef, self.N)


        #Production frontier
        if self.esc == 1:
            self.data["yD"] = 10 * self.data["x1"] ** gamma

        elif self.esc == 2:
            self.data["yD"] = 10 * self.data["x1"]**0.5 * self.data["x2"]**0.3 * self.data["x3"]**0.2
        elif self.esc == 3:
            self.data["yD"] = 10 * self.data["x1"]**0.45 * self.data["x2"]**0.4 * self.data["x3"]**0.15
        elif self.esc == 4:
            self.data["yD"] = 10 * self.data["x1"]**0.6 * self.data["x2"]**0.35 * self.data["x3"]**0.05
        elif self.esc == 5:
            self.data["yD"] = 10 * self.data["x1"] ** 0.65 * self.data["x2"] ** 0.25 * self.data["x3"] ** 0.1
        elif self.esc == 6:
            self.data["yD"] = 10 * self.data["x1"] ** 0.45 * self.data["x2"] ** 0.35 * self.data["x3"] ** 0.1
        elif self.esc == 7:
            self.data["yD"] = 10 * self.data["x1"] ** 0.5 * self.data["x2"] ** 0.25 * self.data["x3"] ** 0.05
        elif self.esc == 8:
            self.data["yD"] = 10 * self.data["x1"] ** 0.4 * self.data["x2"] ** 0.2 * self.data["x3"] ** 0.1
        elif self.esc == 9:
            self.data["yD"] = 10 * self.data["x1"] ** 0.3 * self.data["x2"] ** 0.2 * self.data["x3"] ** 0.15

        if self.esc == 10:
            self.data["yD"] = 10 * self.data["x1"]**0.45 * self.data["x2"]**0.25 * self.data["x3"]**0.15 * self.data["x4"]**0.1 * self.data["x5"]**0.05
        elif self.esc == 11:
            self.data["yD"] = 10 * self.data["x1"]**0.4 * self.data["x2"]**0.25 * self.data["x3"]**0.2 * self.data["x4"]**0.1 * self.data["x5"]**0.05
        elif self.esc == 12:
            self.data["yD"] = 10 * self.data["x1"]**0.35 * self.data["x2"]**0.3 * self.data["x3"]**0.2 * self.data["x4"]**0.1 * self.data["x5"]**0.05
        elif self.esc == 13:
            self.data["yD"] = 10 * self.data["x1"]**0.3 * self.data["x2"]**0.2 * self.data["x3"]**0.15 * self.data["x4"]**0.1 * self.data["x5"]**0.05
        elif self.esc == 14:
            self.data["yD"] = 10 * self.data["x1"]**0.4 * self.data["x2"]**0.2 * self.data["x3"]**0.15 * self.data["x4"]**0.1 * self.data["x5"]**0.05
        elif self.esc == 15:
            self.data["yD"] = 10 * self.data["x1"]**0.25 * self.data["x2"]**0.2 * self.data["x3"]**0.15 * self.data["x4"]**0.1 * self.data["x5"]**0.05
        elif self.esc == 16:
            self.data["yD"] = 10 * self.data["x1"]**0.35 * self.data["x2"]**0.3 * self.data["x3"]**0.15 * self.data["x4"]**0.1 * self.data["x5"]**0.05

        # Inefficiency
        self.data["y"] = self.data["yD"] / (1 + self.u)




