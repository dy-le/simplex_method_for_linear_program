import numpy as np


class simplex_method:
    def __init__(self, A, b, c, B, n):
        self.A = A
        self.c = c
        self.B = B
        self.n = n

        self.xb = np.transpose([b]) 
        self.zn = -self.c[self.n]
        
    def primal_simplex(self, verbor=False):

        count = 0
        Bi = self.A[:,self.B].reshape((-1,len(self.B)))
        N = self.A[:,self.n].reshape((-1, len(self.n)))

        if verbor:
            A_hat = np.concatenate([self.B.T,self.xb.T,N.T,Bi.T]).T
            print("Dictionary\n", A_hat)

        while(np.min(self.zn) < 0):

            j = np.argmin(self.zn)
            ej = np.zeros((1,len(self.zn))).T
            ej[j] = 1

            delta_xb = np.linalg.inv(Bi).dot(N).dot(ej)

            t = np.max(delta_xb/self.xb)**-1

            i = np.argmax(delta_xb/self.xb)
            ei = np.zeros((1,len(self.xb))).T
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

            Bi = self.A[:,self.B].reshape((-1,len(self.B)))
            N = self.A[:,self.n].reshape((-1, len(self.n)))

            count += 1
            optimal = self.xb.T.dot(self.c[self.B]).reshape(-1)[0]

            if verbor:
                A_hat = np.concatenate([self.B.T,self.xb.T,N.T,Bi.T]).T
                print("iter:", count)
                print("Dictionary\n", A_hat)
                print("optimal:", optimal)

        sol = np.zeros(len(self.c))
        sol[self.B] = self.xb

        return {"iter": count,
                "optimal": optimal,
                "sol": sol}

    def dual_simplex(self, verbor=False):

        count = 0
        Bi = self.A[:,self.B].reshape((-1, len(self.B)))
        N  = self.A[:,self.n].reshape((-1, len(self.n)))

        if verbor:
            A_hat = np.concatenate([self.B.T,self.xb.T,N.T,Bi.T]).T
            print("Dictionary\n", A_hat)
        
        while(np.min(self.xb)<0):
            i = np.argmin(self.xb)
            ei = np.zeros((1,len(self.xb))).T
            ei[i] = 1
            
            delta_zn = -(np.linalg.inv(Bi).dot(N)).T.dot(ei)
        
            s = np.max(delta_zn/self.zn)**-1

            j = np.argmax(delta_zn/self.zn)
            ej = np.zeros((1,len(self.zn))).T
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

            Bi = self.A[:,self.B].reshape((-1, len(self.B)))
            N  = self.A[:,self.n].reshape((-1, len(self.n)))

            A_hat = np.concatenate([self.B.T,self.xb.T,N.T,Bi.T]).T

            count += 1
            optimal = self.xb.T.dot(self.c[self.B]).reshape(-1)[0]

            if verbor:
                A_hat = np.concatenate([self.B.T,self.xb.T,N.T,Bi.T]).T
                print("iter:", count)
                print("Dictionary\n", A_hat)
                print("optimal:", optimal)
        
        sol = np.zeros(len(self.c))
        sol[self.B] = self.xb
        
        return {
            "iter": count,
            "optimal": optimal,
            "sol": sol
        }
    
if __name__ == "__main__":

    # inputs 

    # A will contain the coefficients of the constraints 
    A = np.array([[-1,-1, -1, 1, 0],
                [2, -1,  1, 0, 1]])


    # b will contain the amount of resources 
    b = np.array([-2, 1])


    # c will contain coefficients of objective function Z 
    c = np.array([2, -6, 0, 0, 0])

    B = np.array([[3], [4]])
    n = np.array([[0], [1], [2]])

    simplex = simplex_method(A,b,c,B,n)
    simplex.dual_simplex(
        verbor=True
    )