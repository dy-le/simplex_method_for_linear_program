import numpy as numpy

class Simplex_method:
    def __init__(self, A, b, c):
        self.A = A
        self.c = c
        self.B = 0
        self.n = 0
        n_contrain = len(self.A)
        n_var = len(self.c) - n_contrain

        self.B = np.arange(n_var, n_var + n_contrain)[np.newaxis].T
        self.n = np.arange(0, n_var)[np.newaxis].T

        self.xb = np.transpose([b])
        self.zn = -self.c[self.n]
        
        self.status = 'Optimal'
        self.objective = 0

    def solve(self, verbor=False):
        self.count = 0
        for i in self.n:
            if True not in (self.A[:, i] > 0) and self.c[i] > 0:
                print("Unbounded")
                self.status = 'Unbounded'
                
                sol = np.zeros(len(self.c))
                sol[self.B] = self.xb

                return {
                    'status': self.status,
                    "iter": self.count,
                    "objective": self.objective,
                    "sol": sol
                }

        if False not in (self.xb >= 0) and False not in (self.zn <= 0):
            print("Optimal — the problem was trivial")
            sol = np.zeros(len(self.c))
            sol[self.B] = self.xb

            return {
                'status': self.status,
                "iter": self.count,
                "objective": self.objective,
                "sol": sol
            }

        elif False not in (self.xb >= 0) and False in (self.zn <= 0):
            print("primal feasible")
            print("run primal simplex method")
            result = self.primal_simplex(verbor=verbor)

        elif False in (self.xb >= 0) and False not in (self.zn <= 0):
            print("run dual simplex method")
            result = self.solve_two_phase(verbor=verbor)

        else:
            print("dual feasible")
            print("Start convert negative components")
            # self.zn = np.maximum(self.zn, -self.zn)
            # self.zn = np.maximum(self.zn, 0)

            print("run two phase simplex method")
            result = self.solve_two_phase(verbor=verbor)

        return result

    def solve_two_phase(self, verbor=False):
        print("Phase one")
        result = self.dual_simplex(verbor=verbor)

        if result['status'] == 'Infeasible':
            return result

        print("Phase two")
        result = self.primal_simplex(verbor=verbor)

        return result

    def primal_simplex(self, verbor=False):
        objective = -np.inf
        count = 0
        Bi = self.A[:, self.B].reshape((-1, len(self.B)))
        N = self.A[:, self.n].reshape((-1, len(self.n)))

        if verbor:
            A_hat = np.concatenate([self.B.T, self.xb.T, N.T, Bi.T]).T
            print("Objective\n", np.concatenate([self.zn, self.xb]).T)
            print("Dictionary\n", A_hat)

        while(np.min(self.zn) < 0):

            j = np.argmin(self.zn)
            ej = np.zeros((1, len(self.zn))).T
            ej[j] = 1

            delta_xb = np.linalg.inv(Bi).dot(N).dot(ej)

            t = np.max(delta_xb/self.xb)**-1

            if t < 0 or t == np.inf:
                self.status = 'Unbounded'
                sol = np.zeros(len(self.c))
                sol[self.B] = self.xb
                return {
                    'status': self.status,
                    "iter": self.count,
                    "objective": self.objective,
                    "sol": sol
                }

            i = np.argmax(delta_xb/self.xb)
            ei = np.zeros((1, len(self.xb))).T
            ei[i] = 1

            delta_zn = -(np.linalg.inv(Bi).dot(N)).T.dot(ei)
            s = self.zn[j]/delta_zn[j]

            self.xb = self.xb - t*delta_xb
            self.zn = self.zn - s*delta_zn

            self.xb[i] = t
            self.zn[j] = s

            # pivot swap
            pivot = self.B[i].copy()
            self.B[i] = self.n[j].copy()
            self.n[j] = pivot

            Bi = self.A[:, self.B].reshape((-1, len(self.B)))
            N = self.A[:, self.n].reshape((-1, len(self.n)))

            count += 1
            self.count += 1
            self.objective = self.xb.T.dot(self.c[self.B]).reshape(-1)[0]

            if verbor:
                A_hat = np.concatenate([self.B.T, self.xb.T, N.T, Bi.T]).T
                print("iter:", count)
                print("Dictionary\n", A_hat)
                print("objective:", self.objective)

            if self.objective > objective:
                objective = self.objective
            else:
                self.status = 'Infeasible'
                sol = np.zeros(len(self.c))
                sol[self.B] = self.xb
                return {
                    'status': self.status,
                    "iter": self.count,
                    "objective": self.objective,
                    "sol": sol
                }

        sol = np.zeros(len(self.c))
        sol[self.B] = self.xb

        return {
            'status': self.status,
            "iter": self.count,
            "optimal": self.objective,
            "sol": sol
        }

    def dual_simplex(self, verbor=False):

        objective = np.inf
        count = 0
        Bi = self.A[:, self.B].reshape((-1, len(self.B)))
        N = self.A[:, self.n].reshape((-1, len(self.n)))

        if verbor:
            A_hat = np.concatenate([self.B.T, self.xb.T, N.T, Bi.T]).T
            print("Objective\n", np.concatenate([self.zn, self.xb]).T)
            print("Dictionary\n", A_hat)

        while(np.min(self.xb) < 0):
            i = np.argmin(self.xb)
            ei = np.zeros((1, len(self.xb))).T
            ei[i] = 1

            delta_zn = -(np.linalg.inv(Bi).dot(N)).T.dot(ei)

            s = np.max(delta_zn/self.zn)**-1

            j = np.argmax(delta_zn/self.zn)
            ej = np.zeros((1, len(self.zn))).T
            ej[j] = 1

            delta_xb = np.linalg.inv(Bi).dot(N).dot(ej)

            t = self.xb[i]/delta_xb[i]

            self.xb = self.xb - t*delta_xb
            self.zn = self.zn - s*delta_zn

            self.xb[i] = t
            self.zn[j] = s

            # pivot
            pivot = self.B[i].copy()
            self.B[i] = self.n[j].copy()
            self.n[j] = pivot

            Bi = self.A[:, self.B].reshape((-1, len(self.B)))
            N = self.A[:, self.n].reshape((-1, len(self.n)))

            A_hat = np.concatenate([self.B.T, self.xb.T, N.T, Bi.T]).T

            count += 1
            self.count += 1
            self.objective = self.xb.T.dot(self.c[self.B]).reshape(-1)[0]

            if verbor:
                A_hat = np.concatenate([self.B.T, self.xb.T, N.T, Bi.T]).T
                print("iter:", count)
                print("Dictionary\n", A_hat)
                print("objective:", self.objective)

            if self.objective < objective:
                objective = self.objective
            else:
                self.status = 'Infeasible'
                sol = np.zeros(len(self.c))
                sol[self.B] = self.xb
                return {
                    'status': self.status,
                    "iter": self.count,
                    "objective": self.objective,
                    "sol": sol
                }

        sol = np.zeros(len(self.c))
        sol[self.B] = self.xb

        return {
            'status': self.status,
            "iter": self.count,
            "objective": self.objective,
            "sol": sol
        }

if if __name__ == "__main__":
    print("Exercise 2.3")
    # A will contain the coefficients of the constraints
    A = np.array([[-1, -1, -1, 1, 0],
                [2, -1,  1, 0, 1]])

    # b will contain the amount of resources
    b = np.array([-2, 1])

    # c will contain coefficients of objective function Z
    c = np.array([2, -6, 0, 0, 0])

    simplex = Simplex_method(A, b, c)

    print(simplex.solve(verbor=True))
