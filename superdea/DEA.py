import pandas as pd
import math
INF = math.inf
from docplex.mp.model import Model
from itertools import combinations


class DEA:
    def __init__(self, matrix, x, y):
        self.xCol = x
        self.yCol = y
        self.matrix = matrix.loc[:, x + y]  # Order variables
        self.x = matrix.columns.get_indexer(x).tolist()  # Index var.ind in matrix
        self.y = matrix.columns.get_indexer(y).tolist()  # Index var. obj in matrix
        self.N = len(self.matrix)
        self.nX = len(self.x)
        self.nY = len(self.y)

    'Destructor'
    def __del__(self):
        try:
            del self.N
            del self.matrix
            del self.nX
            del self.nY
            del self.x
            del self.y
            del self.xCol
            del self.yCol

        except Exception:
            pass

    def predict(self):
        for i in range(self.N):
            pred = self._predictor(self.matrix.iloc[i, self.x])
            for j in range(self.nY):
                self.matrix.loc[i, "p_" + str(self.yCol[j])] = pred[j]
        return self.matrix

    def _predictor(self, XArray):
        yMax = [-INF] * self.nY
        for n in range(self.N):
            newMax = True
            for i in range(len(XArray)):
                for j in range(self.nY):
                    if i < self.y[j]:
                        if self.matrix.iloc[n, i] > XArray[i]:
                            newMax = False
                            break
            for j in range(self.nY):
                if newMax and yMax[j] < self.matrix.iloc[n, self.y[j]]:
                    yMax[j] = self.matrix.iloc[n, self.y[j]]

        return yMax

    def _DEA(self, x, y):
        # Prepare matrix
        self.xmatrix = self.matrix.iloc[:, self.x].T  # xmatrix
        self.ymatrix = self.matrix.iloc[:, self.y].T  # ymatrix

        # create one model instance, with a name
        m = Model(name='DEA')

        # by default, all variables in Docplex have a lower bound of 0 and infinite upper bound
        beta = {0: m.continuous_var(name="beta")}

        # Constrain 2.4
        name_lambda = {i: m.continuous_var(name="l_{0}".format(i)) for i in range(self.N)}

        # Constrain 2.3
        m.add_constraint(m.sum(name_lambda[n] for n in range(self.N)) == 1)  # sum(lambda) = 1

        # Constrain 2.1 y 2.2
        for i in range(self.nX):
            # Constrain 2.1
            m.add_constraint(m.sum(self.xmatrix.iloc[i, j] * name_lambda[j] for j in range(self.N)) <= x[i])

        for i in range(self.nY):
            # Constrain 2.2
            m.add_constraint(
                m.sum(self.ymatrix.iloc[i, j] * name_lambda[j] for j in range(self.N)) >= beta[0] * y[i])

        # objetive
        m.maximize(beta[0])

        # Model Information
        # m.print_information()

        m.solve()

        # Solución
        if m.solution == None:
            sol = 0
        else:
            sol = m.solution.objective_value
        return sol

    def DEA(self):
        nameCol = "DEA"
        matrix = self.matrix.copy()
        matrix.loc[:, nameCol] = 0

        for i in range(len(matrix)):
            matrix.loc[i, nameCol] = self._DEA(self.matrix.iloc[i, self.x].to_list(),
                                                                    self.matrix.iloc[i, self.y].to_list())
        # Caso mono-output. Predicción
        if self.nY == 1:
            # Predicción
            matrix.loc[:, "yDEA"] = matrix.loc[:, "DEA"] * matrix.loc[:, "y"]
        elif self.nY > 1:
            # Create data frame
            for j in range(self.nY):
                matrix.loc[:, "yDEA{0}".format(self.yCol[j])] = 0
            # Calculate predictions
            for j in range(self.nY):
                matrix.loc[:, "yDEA{0}".format(self.yCol[j])] = matrix.loc[:, "DEA"] * matrix.loc[:, self.yCol[j]]

        return matrix

    def _superDEA(self, matrix, x_ind, y_ind, x, y, ind): #ind = index del dato a eliminar
        nX = len(x_ind)
        # Prepare matrix
        self.xmatrix = matrix.drop([ind], axis=0).reset_index(drop=True)
        self.xmatrix = self.xmatrix.loc[:, x_ind].T  # xmatrix
        self.ymatrix = matrix.drop([ind], axis=0).reset_index(drop=True)
        self.ymatrix = self.ymatrix.loc[:, y_ind] # ymatrix


        # create one model instance, with a name
        m = Model(name='superDEA')

        # by default, all variables in Docplex have a lower bound of 0 and infinite upper bound
        beta = {0: m.continuous_var(name="beta")}

        # Constrain 2.4
        name_lambda = {i: m.continuous_var(name="l_{0}".format(i)) for i in range(self.N-1)}

        # Constrain 2.3
        m.add_constraint(m.sum(name_lambda[n] for n in range(self.N-1)) == 1)  # sum(lambda) = 1

        # Constrain 2.1 y 2.2
        for i in range(nX):
            # Constrain 2.1
            m.add_constraint(m.sum(self.xmatrix.iloc[i, j] * name_lambda[j] for j in range(self.N - 1)) <= x[i])

        for i in range(self.nY):
            # Constrain 2.2
            m.add_constraint(
                m.sum(self.ymatrix.iloc[j] * name_lambda[j] for j in range(self.N - 1)) >= beta[0] * y[i])

        # objetive
        m.maximize(beta[0])

        # Model Information
        # m.print_information()

        m.solve()

        # Solución
        if m.solution == None:
            sol = None
        else:
            sol = m.solution.objective_value
        return sol

    def _supDEA(self, matrix, x):
        nameCol = "superDEA"
        matrix.loc[:, nameCol] = 0

        for i in range(len(matrix)):
            matrix.loc[i, nameCol] = self._superDEA(matrix, x, "y", matrix.loc[i, x].to_list(),
                                                    [matrix.loc[i, "y"]], i)
        # Caso mono-output. Predicción
        if self.nY == 1:
            # Predicción
            matrix.loc[:, "ysuperDEA"] = matrix.loc[:, "superDEA"] * matrix.loc[:, "y"]
        elif self.nY > 1:
            # Create data frame
            for j in range(self.nY):
                matrix.loc[:, "ysuperDEA{0}".format(self.yCol[j])] = 0
            # Calculate predictions
            for j in range(self.nY):
                matrix.loc[:, "ysuperDEA{0}".format(self.yCol[j])] = matrix.loc[:, "superDEA"] * matrix.loc[:, self.yCol[j]]

        return matrix

    def mse(self, obs, pred):
        err = 0
        for i in range(len(pred)):
            # Mono output
            if self.nY == 1:
                err += (obs[i] - pred[i]) ** 2
        return err / len(pred)

    def superDEA(self):
        min_err = INF
        output = sum([list(map(list, combinations(self.xCol, i))) for i in range(len(self.xCol) + 1)], [])
        # Combinación de todas las X's
        for xi in range(1, len(output)):
            print("Inputs ", output[xi])
            matrix = self.matrix.copy()
            matrix = matrix[output[xi] + self.yCol]
            supDEA = self._supDEA(matrix, output[xi])

            mse = (((matrix.loc[:, "y"] - matrix.loc[:, "ysuperDEA"])**2).sum())/self.N
            print(mse)
            if mse < min_err:
                min_err = mse
                result_supDEA = supDEA.copy()
                result_output = output[xi]

        return result_supDEA, result_output




class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

class EXIT(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return style.YELLOW + "\n\n" + self.message + style.RESET
