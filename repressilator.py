import numpy as np
from scipy.integrate import solve_ivp

eps = 1e-12

# ----------------------------------------------------------------------------------------------------------------------
class Repressilator(object):
    def __init__(self, y_real, params):
        super().__init__()
        self.y_real = y_real
        self.params = params

    def repressilator_model(self, t, y):

        m1, m2, m3, p1, p2, p3 = y[0], y[1], y[2], y[3], y[4], y[5]

        alpha0 = self.params['alpha0']
        n = self.params['n']
        beta = self.params['beta']
        alpha = self.params['alpha']

        dm1_dt = -m1 + alpha / (1. + p3**n + eps) + alpha0
        dp1_dt = -beta * (p1 - m1)
        dm2_dt = -m2 + alpha / (1. + p1**n + eps) + alpha0
        dp2_dt = -beta * (p2 - m2)
        dm3_dt = -m3 + alpha / (1. + p2**n + eps) + alpha0
        dp3_dt = -beta * (p3 - m3)

        return dm1_dt, dm2_dt, dm3_dt, dp1_dt, dp2_dt, dp3_dt

    def solve_repressilator(self):
        # we need to use lambda function if we want to pass some parameters
        solution = solve_ivp(lambda t, y: self.repressilator_model(t, y),
                             t_span=(self.params['t0'], self.params['t1']), y0=self.params['y0'],
                             method='RK45', t_eval=self.params['t_points'])
        y_points = np.asarray(solution.y)
        return self.params['t_points'], y_points

    @staticmethod
    def loss(y_real, y_model):
        # we assume only m's are observed!
        y_r = y_real[0:3]
        y_m = y_model[0:3]
        if y_r.shape[1] == y_m.shape[1]:
            return np.mean(np.sqrt(np.sum((y_r - y_m)**2, 0)))
        else:
            return np.infty

    def objective(self, x):
        if len(x.shape) > 1:
            objective_values = []
            for i in range(x.shape[0]):
                xi = x[i]
                self.params['alpha0'] = xi[0]
                self.params['n'] = xi[1]
                self.params['beta'] = xi[2]
                self.params['alpha'] = xi[3]

                _, y_model = self.solve_repressilator()
                objective_values.append(self.loss(self.y_real, y_model))

            objective_values = np.asarray(objective_values)
        else:
            self.params['alpha0'] = x[0]
            params['n'] = x[1]
            params['beta'] = x[2]
            params['alpha'] = x[3]

            _, y_model = self.solve_repressilator()
            objective_values = self.loss(self.y_real, y_model)

        return objective_values

# EXAMPLE FOR LOADING DATA & PARAMS, AND CALCULATING OBJECTIVE
if __name__ == "__main__":
    import pickle
    import matplotlib.pyplot as plt

    # loading data & params
    y_real = pickle.load(open('data.pkl', 'rb'))
    params = pickle.load(open('params.pkl', 'rb'))

    print(params)

    plt.plot(params['t_points'], y_real[0])
    plt.plot(params['t_points'], y_real[1])
    plt.plot(params['t_points'], y_real[2])
    plt.show()

    # calculating objective
    r = Repressilator(y_real, params)

    # if a single individual
    x = np.random.uniform(low=0.1, high=10., size=(4,))
    print(r.objective(x))

    # if a batch of individuals (population)
    x = np.random.uniform(low=0.1, high=10., size=(10,4))
    print(r.objective(x))