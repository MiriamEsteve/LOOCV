import pandas as pd
import numpy as np
from docplex.mp.model import Model
import math
from sklearn.linear_model import LinearRegression
import statsmodels.api as sma
from scipy import stats
from scipy.stats import t

INF = math.inf


class Ruggiero:
    def __init__(self, matrix, VAR, y, model): # x = Conjunto de variables que siempre van a estar; model = "CCR" excepto escenario 11 que es BCC
        self.model = model
        self.VARCol = VAR
        self.noVARCol = list(matrix.drop(self.VARCol + y, axis=1).columns)
        self.xCol = self.VARCol + self.noVARCol
        self.yCol = y
        self._check_enter_parameters(matrix, self.xCol, y)

        # Matrix original
        self.matrixOri = matrix.loc[:, self.xCol + self.yCol]  # Order variables
        self.NOri = len(self.matrixOri)
        # Para CV
        self.matrix = self.matrixOri.copy()
        self.N = self.NOri

        self.VAR = matrix.columns.get_indexer(self.VARCol).tolist()  # Index var.ind in matrix
        self.noVAR = matrix.columns.get_indexer(self.noVARCol).tolist()  # Index var.ind in matrix
        self.y = matrix.columns.get_indexer(self.yCol).tolist()  # Index var. obj in matrix
        self.mVAR = len(self.VAR)
        self.mnoVAR = len(self.noVAR)
        self.s = len(self.y)

    'Destructor'
    def __del__(self):
        try:
            del self.N
            del self.matrix
            del self.mVAR
            del self.mnoVAR
            del self.s
            del self.VAR
            del self.noVAR
            del self.VARCol
            del self.noVARCol
            del self.y
            del self.xCol
            del self.yCol

        except Exception:
            pass

    def _check_enter_parameters(self, matrix, x, y):
        # var. x and var. y have been procesed
        if type(x[0]) == int or type(y[0]) == int:
            self.matrix = matrix
            if any(self.matrix.dtypes == 'int64'):
                self.matrix = self.matrix.astype('float64')
            return
        else:
            self.matrix = matrix.loc[:, x + y]  # Order variables
            if any(self.matrix.dtypes == 'int64'):
                self.matrix = self.matrix.astype('float64')

        if len(matrix) == 0:
            raise EXIT("ERROR. The dataset must contain data")
        elif len(x) == 0:
            raise EXIT("ERROR. The inputs of dataset must contain data")
        elif len(y) == 0:
            raise EXIT("ERROR. The outputs of dataset must contain data")
        else:
            cols = x + y
            for col in cols:
                if col not in matrix.columns.tolist():
                    raise EXIT("ERROR. The names of the inputs or outputs are not in the dataset")

            for col in x:
                if col in y:
                    raise EXIT("ERROR. The names of the inputs and the outputs are overlapping")
        '''
        if len(matrix) > 10 and (nu < (1/len(matrix)) or nu > 0.1):
            raise EXIT("ERROR. The nu hiperparamether must be more than 1/n and less than 0.1")
        if len(matrix) < 10 and (nu < 0.1 or nu > 0.5):
            raise EXIT("ERROR. The nu hiperparamether must be more than 0.1 and less than 0.5")
        '''

    def fit(self):
        cont = 0
        while self.mnoVAR != 0 and cont != self.mnoVAR: #Si no hay más noVAR que probar o si no hay más significativas
            # Resolver DEA orientación output con las inputs seleccionadas. (VAR)
            dataDEA = self.DEA(self.matrix.copy())

            X = self.matrix.iloc[:, self.noVAR]
            y = dataDEA["DEA"] # Scores = 1/ TE_i

            '''
            mod = sma.OLS(y, X)
            fii = mod.fit()
            p_values = fii.pvalues
            beta = fii.params
            '''
            '''
            lm = LinearRegression()
            lm.fit(X, y)

            predictions = lm.predict(X)
            params = np.append(lm.intercept_, lm.coef_)

            newX = pd.DataFrame({"Constant": np.ones(len(X))}).join(pd.DataFrame(X))
            MSE = (sum((y - predictions) ** 2)) / (len(newX) - len(newX.columns))

            var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
            sd_b = np.sqrt(var_b)
            ts_b = params / sd_b
            p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - len(newX.columns) - 1))) for i in ts_b]
            '''
            reg = LinearRegression().fit(X, y)
            beta = [reg.intercept_] + reg.coef_.tolist()

            # compute the p-values
            # add ones column
            X1 = np.column_stack((np.ones(self.N), X))
            # standard deviation of the noise.
            sigma_hat = np.sqrt(np.sum(np.square(y - X1.beta)) / (self.N - X1.shape[1])) # X1@beta
            # estimate the covariance matrix for beta
            beta_cov = np.linalg.inv(X1.T.X1) # X1.T@X1
            # the t-test statistic for each variable from the formula from above figure
            t_vals = beta / (sigma_hat * np.sqrt(np.diagonal(beta_cov)))
            # compute 2-sided p-values.
            p_values = t.sf(np.abs(t_vals), self.N - X1.shape[1]) * 2

            cont = 0 # Contar no significativas. Si es = mnoVAR paramos.
            for i in range(1, self.mnoVAR + 1):
                if beta[i] > 0 and p_values[i] < 0.1:
                    var = self.noVAR.pop(0)
                    print("Select: ", self.xCol[var])
                    self.VAR.append(var) # Introducir variable en VAR y eliminarla de noVAR
                    self.VARCol.append(self.xCol[var]) # Introducir variable en VAR y eliminarla de noVAR
                    self.mnoVAR = len(self.noVAR)
                    self.mVAR = len(self.VAR)
                else:
                    cont += 1
            self.mnoVAR = len(self.noVAR)
            self.mVAR = len(self.VAR)

        # Modelo DEA con las variables seleccionadas
        dataDEA = self.DEA(self.matrix.copy())

        return dataDEA

    ######################################################## DEA #######################################################
    def _scoreDEA_BCC_output(self, x, y):
        # Prepare matrix
        self.xmatrix = self.matrix.iloc[:, self.VAR].T  # xmatrix
        self.ymatrix = self.matrix.iloc[:, self.y].T  # ymatrix

        # create one model instance, with a name
        m = Model(name='beta_DEA_BCC')

        # by default, all variables in Docplex have a lower bound of 0 and infinite upper bound
        beta = {0: m.continuous_var(name="beta")}

        # Constrain 2.4
        name_lambda = {i: m.continuous_var(name="l_{0}".format(i)) for i in range(self.N)}

        # Constrain 2.3
        m.add_constraint(m.sum(name_lambda[n] for n in range(self.N)) == 1)  # sum(lambda) = 1

        # Constrain 2.1 y 2.2
        for i in range(self.mVAR):
            # Constrain 2.1
            m.add_constraint(m.sum(self.xmatrix.iloc[i, j] * name_lambda[j] for j in range(self.N)) <= x[i])

        for i in range(self.s):
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
            sol = None
        else:
            sol = m.solution.objective_value
        return sol

    def _scoreDEA_CCR_output(self, x, y):
        # Prepare matrix
        self.xmatrix = self.matrix.iloc[:, self.VAR].T  # xmatrix
        self.ymatrix = self.matrix.iloc[:, self.y].T  # ymatrix

        # create one model instance, with a name
        m = Model(name='beta_DEA_CCR')

        # by default, all variables in Docplex have a lower bound of 0 and infinite upper bound
        beta = {0: m.continuous_var(name="beta")}

        # Constrain 2.4
        name_lambda = {i: m.continuous_var(name="l_{0}".format(i)) for i in range(self.N)}

        # Constrain 2.3
        #m.add_constraint(m.sum(name_lambda[n] for n in range(self.N)) == 1)  # sum(lambda) = 1

        # Constrain 2.1 y 2.2
        for i in range(self.mVAR):
            # Constrain 2.1
            m.add_constraint(m.sum(self.xmatrix.iloc[i, j] * name_lambda[j] for j in range(self.N)) <= x[i])

        for i in range(self.s):
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
            sol = None
        else:
            sol = m.solution.objective_value
        return sol

    def DEA(self, matrix):
        nameCol = "DEA"
        # matrix = self.matrixOri.copy()
        matrix.loc[:, nameCol] = 0

        for i in range(len(matrix)):
            if self.model == "BCC":
                matrix.loc[i, nameCol] = self._scoreDEA_BCC_output(self.matrixOri.iloc[i, self.VAR].to_list(),
                                                                    self.matrixOri.iloc[i, self.y].to_list())
            elif self.model == "CCR":
                matrix.loc[i, nameCol] = self._scoreDEA_CCR_output(self.matrixOri.iloc[i, self.VAR].to_list(),
                                                                   self.matrixOri.iloc[i, self.y].to_list())
        # Caso mono-output. Predicción
        if self.s == 1:
            # Predicción
            matrix.loc[:, "DEA"] = (1 / matrix.loc[:, "DEA"])
            matrix.loc[:, "yDEA"] = matrix.loc[:, "DEA"] * matrix.loc[:, "y"]
        elif self.s > 1:
            # Create data frame
            for j in range(self.s):
                matrix.loc[:, "yDEA{0}".format(self.yCol[j])] = 0
            # Calculate predictions
            for j in range(self.s):
                matrix.loc[:, "yDEA{0}".format(self.yCol[j])] = matrix.loc[:, "DEA"] * matrix.loc[:, self.yCol[j]]

        return matrix

    def MSE_err_DEA(self, matrix, dataDEA):
        err = 0
        for i in range(len(matrix)):
            # Mono output
            if self.s == 1:
                err += (matrix.iloc[i, self.y[0]] - dataDEA.loc[i, "DEA"]) ** 2
            else:
                for j in range(self.s):
                    err += (matrix.iloc[i, self.y[j]] - dataDEA.loc[i, "yDEA{0}".format(self.yCol[j])]) ** 2

        return err / len(matrix)

    ################################### ERROR with Theorical Frontier ##################################################
    def MSE_theoric(self, matrix, matrix2, dea_or_udea):
        err = 0
        for i in range(len(matrix)):
            err += (matrix.loc[i, "yD"] - matrix2.loc[i, dea_or_udea]) ** 2
        return err / len(matrix)

    def _fi_Theoric(self, x, y):
        # ---------------------- z = ln(y2, y1) ------------------------------------
        z = np.log(y[1] / y[0])

        # -------------- Pasos 2 y 3 para obtener y1*, y2* -------------------------
        # Ln de x1 y x2
        ln_x1 = np.log(x[0])
        ln_x2 = np.log(x[1])
        # Operaciones para ln_y1_ast
        op1 = -1 + 0.5 * z + 0.25 * (z ** 2) - 1.5 * ln_x1
        op2 = -0.6 * ln_x2 + 0.2 * (ln_x1 ** 2) + 0.05 * (ln_x2 ** 2) - 0.1 * ln_x1 * ln_x2
        op3 = 0.05 * ln_x1 * z - 0.05 * ln_x2 * z
        ln_y1_ast = -(op1 + op2 + op3)

        # Y de ese valor determinamos y1*=exp(ln(y1*))
        y1_ast = np.exp(ln_y1_ast)
        # P3(Calculamos ln(y2*) como z + ln(y1*). Del ln(y2*), sacamos y2* = exp(ln(y2*))
        # y2_ast = np.exp(ln_y1_ast + z)

        # ------------------ Obtener fi --------------------------------------------
        fi_y1 = y1_ast / y[0]
        # fi_y2 = y2_ast / y[1]

        return fi_y1

    def fit_Theoric(self):
        nameCol = "yD"
        self.matrix.loc[:, nameCol] = 0

        for i in range(len(self.matrix)):
            self.matrix.loc[i, nameCol] = self._fi_Theoric(self.matrix.iloc[i, self.x].to_list(),
                                                           self.matrix.iloc[i, self.y].to_list())

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


