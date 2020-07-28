import numpy as np
import cplex

def cplex_lib(A, b, c):
    # Input all the data and parameters here
    num_constraints = len(A)
    num_decision_var = len(c) - num_constraints
    
    n = np.arange(0, num_decision_var)[np.newaxis].T

    A = A[:,n.T].reshape(num_constraints, num_decision_var).tolist()
    b = b.tolist()
    c = c[n].T.reshape(len(n)).tolist()

    # constraint_type = ["L", "L", "L"] # Less, Greater, Equal
    constraint_type = ["L"]*num_constraints
    # ============================================================

    # Establish the Linear Programming Model
    myProblem = cplex.Cplex()

    # Add the decision variables and set their lower bound and upper bound (if necessary)
    myProblem.variables.add(names= ["x"+str(i) for i in range(num_decision_var)])
    for i in range(num_decision_var):
        myProblem.variables.set_lower_bounds(i, 0.0)

    # Add constraints
    for i in range(num_constraints):
        myProblem.linear_constraints.add(
            lin_expr= [cplex.SparsePair(ind= [j for j in range(num_decision_var)], val= A[i])],
            rhs= [b[i]],
            names = ["c"+str(i)],
            senses = [constraint_type[i]]
        )

    # Add objective function and set its sense
    for i in range(num_decision_var):
        myProblem.objective.set_linear([(i, c[i])])
    myProblem.objective.set_sense(myProblem.objective.sense.maximize)

    # Solve the model and print the answer
    myProblem.solve()
    return{
        'objective': myProblem.solution.get_objective_value(),
        'status': myProblem.solution.get_status_string(),
        'sol': myProblem.solution.get_values()
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

    cplex_lib(A, b, c)