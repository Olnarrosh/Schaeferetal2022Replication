
import numpy as np

from sklearn.linear_model import LinearRegression

#unabh. Variablen: spearman ähnlichkeit, Korpsugrößetraining(#Sätze), verhältnisclaimnichtclaimquell (anzahlclaims/anzahlsätzegesamt) auf 3 kommastellen, verhältnisclaimnichtclaimziel,
# usdebtrain, microtrain, essaytrain, cmvtrain, mardytrain  (--> alle binary)

# für leave one out: korpusgröße = summe aller größen, verhältnisclaims = summe aller claims / gesamtkorpusgröße

# also Länge pro liste ist 9

# Anzahl Listen = Anzahl Experimente: 5 + 20 + 5 = 30
if __name__ == "__main__":
                    # 5 in domain
                    # TODO: alle leave one out, alle spearman, alle korpusgröße mardy, claimverhältnisse mardy
    X = np.array([[0, 29621, 0.481, 0.481, 1, 0, 0, 0, 0],
                  [0, 965, 0.035, 0.035, 0, 1, 0, 0, 0],
                  [0, 1665, 0.305, 0.305, 0, 0, 1, 0, 0],
                  [0, 2798, 0.366, 0.366, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1],
                  # 5 leave one out
                  [0, 0, 0, 0, 0, 1, 1, 1, 1],
                  [0, 0, 0, 0, 1, 0, 1, 1, 1],
                  [0, 0, 0, 0, 1, 1, 0, 1, 1],
                  [0, 0, 0, 0, 1, 1, 1, 0, 1],
                  [0, 35049, 0.451, 0, 1, 1, 1, 1, 0],
                  # 4 train usdeb
                  [0, 29621, 0.481, 0.035, 1, 0, 0, 0, 0],
                  [0, 29621, 0.481, 0.305, 1, 0, 0, 0, 0],
                  [0, 29621, 0.481, 0.366, 1, 0, 0, 0, 0],
                  [0, 29621, 0.481, 0, 1, 0, 0, 0, 0],
                  #4 train micro
                  [0, 965, 0.035, 0.481, 0, 1, 0, 0, 0],
                  [0, 965, 0.035, 0.305, 0, 1, 0, 0, 0],
                  [0, 965, 0.035, 0.366, 0, 1, 0, 0, 0],
                  [0, 965, 0.035, 0, 0, 1, 0, 0, 0],
                  # 4 train essay
                  [0, 1665, 0.305, 0.481, 0, 0, 1, 0, 0],
                  [0, 1665, 0.305, 0.035, 0, 0, 1, 0, 0],
                  [0, 1665, 0.305, 0.366, 0, 0, 1, 0, 0],
                  [0, 1665, 0.305, 0, 0, 0, 1, 0, 0],
                  #4 train cmv
                  [0, 2798, 0.366, 0.481, 0, 0, 0, 1, 0],
                  [0, 2798, 0.366, 0.035, 0, 0, 0, 1, 0],
                  [0, 2798, 0.366, 0.305, 0, 0, 0, 1, 0],
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

