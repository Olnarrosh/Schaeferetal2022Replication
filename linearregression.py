
import numpy as np

from sklearn.linear_model import LinearRegression

#unabh. Variablen: spearman ähnlichkeit (3 kommastellen), Korpsugrößetraining(#Sätze), verhältnisclaimnichtclaimquell (anzahlclaims/anzahlsätzegesamt) auf 3 kommastellen, verhältnisclaimnichtclaimziel,
# usdebtrain, microtrain, essaytrain, cmvtrain, mardytrain  (--> alle binary)

# für leave one out: korpusgröße = summe aller größen, verhältnisclaims = summe aller claims / gesamtkorpusgröße

# also Länge pro liste ist 9
# spearman: cmv essay 0.209, cmv micro 0.249, micro essay 0.302, usdeb cmv 0.211, usdeb essay 0.281, usdeb micro 0.206
# Anzahl Listen = Anzahl Experimente: 5 + 20 + 5 = 30
if __name__ == "__main__":
                    # 5 in domain
                    # TODO: alle leave one out, alle spearman mit mardy, alle korpusgröße mardy, alle claimverhältnisse mardy
    X = np.array([[1, 29621, 0.481, 0.481, 1, 0, 0, 0, 0],
                  [1, 965, 0.035, 0.035, 0, 1, 0, 0, 0],
                  [1, 1665, 0.305, 0.305, 0, 0, 1, 0, 0],
                  [1, 2798, 0.366, 0.366, 0, 0, 0, 1, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                  # 5 leave one out
                  [0, 0, 0, 0, 0, 1, 1, 1, 1],
                  [0, 0, 0, 0, 1, 0, 1, 1, 1],
                  [0, 0, 0, 0, 1, 1, 0, 1, 1],
                  [0, 0, 0, 0, 1, 1, 1, 0, 1],
                  [0, 35049, 0.451, 0, 1, 1, 1, 1, 0],
                  # 4 train usdeb
                  [0.206, 29621, 0.481, 0.035, 1, 0, 0, 0, 0],
                  [0.281, 29621, 0.481, 0.305, 1, 0, 0, 0, 0],
                  [0.211, 29621, 0.481, 0.366, 1, 0, 0, 0, 0],
                  [0, 29621, 0.481, 0, 1, 0, 0, 0, 0],
                  #4 train micro
                  [0.206, 965, 0.035, 0.481, 0, 1, 0, 0, 0],
                  [0.302, 965, 0.035, 0.305, 0, 1, 0, 0, 0],
                  [0.249, 965, 0.035, 0.366, 0, 1, 0, 0, 0],
                  [0, 965, 0.035, 0, 0, 1, 0, 0, 0],
                  # 4 train essay
                  [0.281, 1665, 0.305, 0.481, 0, 0, 1, 0, 0],
                  [0.302, 1665, 0.305, 0.035, 0, 0, 1, 0, 0],
                  [0.209, 1665, 0.305, 0.366, 0, 0, 1, 0, 0],
                  [0, 1665, 0.305, 0, 0, 0, 1, 0, 0],
                  #4 train cmv
                  [0.211, 2798, 0.366, 0.481, 0, 0, 0, 1, 0],
                  [0.249, 2798, 0.366, 0.035, 0, 0, 0, 1, 0],
                  [0.209, 2798, 0.366, 0.305, 0, 0, 0, 1, 0],
                  [0, 2798, 0.366, 0, 0, 0, 0, 1, 0],
                  # 4 train mardy
                  [0, 0, 0, 0.481, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0.035, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0.305, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0.366, 0, 0, 0, 0, 1]])
    y = [] #TODO insert all F-Scores, auf 3 kommastellen runden
    reg = LinearRegression().fit(X, y)
    # prints all weights -> see which independent variables are most important!!
    print(reg.coef_)

