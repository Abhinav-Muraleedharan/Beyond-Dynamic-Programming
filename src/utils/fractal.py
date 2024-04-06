"""
Class containing definition of a Fractal function 
and methods to compute optima of Fractal functions
"""
import random

class Fractal:

    def __init__(self,alpha_0,alpha_1,coefficients):

        """

        Define Faber Schauder Coefficients of Fractal function (Section 4.1 of https://arxiv.org/pdf/2306.15029.pdf )
        self.alpha_0 -> alpha_0 in the Faber Schauder Expansion  (Eq 21)
        self.a_1 -> alpha_1 (Eq 22)
        self.coefficients -> Rest of the coefficients alpha_ij

        """
        self.alpha_0 = alpha_0 
        self.alpha_1 = alpha_1
        self.coefficients = coefficients 

    def _derivative_mod_x(a,b,x):

        """
        
        Function to compute derivative of |ax - b|

        """
        if x == b/a:
            derivative = -a
        else:
            derivative = a*(abs(a*x - b)/(a*x - b))
        return derivative
    
    def _d_S_i_j(self,l,i,j):
        """

        Function to estimate derivative of the term e_ij in the Faber Schauder Expansion

        """
        derivative = (2**j)*(self._derivative_mod_x(1,(i/(2**j)),l) + self._derivative_mod_x(1,((i+1)/(2**j)),l) - self._derivative_mod_x(2,((2*i+1)/(2**j)),l))
        return derivative


    def _grad_fractal(self,l):
        """

        Function to estimate derivative of Fractal function at any input value l 

        """
        grad_f = self.alpha_1
        j_max = len(self.coefficients)
        j = 0
        while j < j_max:
            i = 0
            while i <=2**j - 1:
                grad_f = grad_f + self._d_S_i_j(l,i,j)*self.coefficients[j][i]
                i = i + 1
            j = j + 1
        return grad_f

    def compute_optima_gradient_descent(self):

        """ 
        
        Compute Optima of fractal function using Algorithm described in the Bdp paper
        
        """
        max_iter = 100
        i = 0
        lr = 0.01 # learning rate
        grad_prev = 0
        l_array = []
        grad_array = []
        i_array = []
        while i < max_iter:

            if i == 0:
                # random initialization 
                l = random.random()
            grad = self._grad_fractal(self,l)
            l = l - grad*lr
            lr = lr*(2**(-i))
            grad_sq = grad**2
            grad_array.append(grad_sq)
            l_array.append(l)
            i_array.append(i)

            if grad*grad_prev < 0:
                if grad**2 < 0.01:
                    break

            grad_prev = grad

            if l < 0:
                l = 0
                break

            if l > 1:
                l = 0.9999999
                break

            i = i +1

        return l 

