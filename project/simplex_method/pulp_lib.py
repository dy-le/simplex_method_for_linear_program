import pulp as p
import numpy as np

def pulp_lib(A, b, c, verbor=False):
    # Generate B and N
    n_contrain = len(A)
    n_var = len(c) - n_contrain

    B = np.arange(n_var, n_var + n_contrain)[np.newaxis].T
    n = np.arange(0, n_var)[np.newaxis].T

    # Create a LP Minimization problem 
    Lp_prob = p.LpProblem('Problem', p.LpMaximize) 

    # Create problem Variables
    x = [p.LpVariable("x"+str(i), lowBound = 0) for i in range(1,n_var+1)]

    # Objective Function 
    objective = 0
    for i in range(n_var):
        objective += c[i]*x[i]

    Lp_prob += objective 

    # Constraints:
    for i in range(n_contrain):
        contrain = 0
        for j in range(n_var):
            contrain += A[i,j]*x[j] <= b[i]/n_var
        Lp_prob += contrain

    status = Lp_prob.solve() # Solver 
    
    if verbor:
        print(p.LpStatus[status]) # The solution status 
        # Printing the final solution 
        print(p.value(Lp_prob.objective))
    
    return {
        'status': p.LpStatus[status],
        'objective': p.value(Lp_prob.objective)
    }

if __name__ == "__main__":
    print("Exercise 2.3")
    # A will contain the coefficients of the constraints
    A = np.array([[-1, -1, -1, 1, 0],
                  [2, -1,  1, 0, 1]])

    # b will contain the amount of resources
    b = np.array([-2, 1])

    # c will contain coefficients of objective function Z
    c = np.array([2, -6, 0, 0, 0])

    pulp_lib(A, b, c, verbor=False)