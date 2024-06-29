import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from flask import Flask, request, render_template
from io import BytesIO
import base64



def pfr_calculator(flowrate, K, concA, concB, concC, concD, a, b, c, d, X, state):
    x = np.linspace(0.1, 0.8, 100)
    v = flowrate
    c = concA
    thetaB = concB / concA
    thetaC = concC / concA
    thetaD = concD / concA

    if state == 'Liquid':
        Ca = c * (1 - x)
        Cb = c * (thetaB - (b / a) * x)
        Cc = c * (thetaC + (c / a) * x)
        Cd = c * (thetaD + (d / a) * x)
        rA = K * (c ** a) * ((1 - x) ** a) * ((thetaB - (b / a) * x) ** b)
        V = rA

        for i in range(len(x)):
            def integrand(t):
                return (c * v) / (K * (c ** a) * ((1 - t) ** a) * ((thetaB - (b / a) * t) ** b))
            V[i] = quad(integrand, 0, x[i])[0]
        
        for i in range(len(X)):
            def integrand(t):
                return (c*v)/ ((c ** (a + b)) * ((1 - t) ** a) * ((thetaB - (b / a) * t) ** b))
            V1[i] = quad(integrand, 0, x[i])[0]
        print (V1)
        

        # Plot results
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        ax1.plot(x, -(concA/ rA), 'b-')
        ax1.set_title('Levenspiel Plot (PFR)')
        ax1.set_xlabel('x')
        ax1.set_ylabel('-Ca0*V0/ rA (%)')

        ax2.plot(V, Ca, 'r-', label='a')
        ax2.plot(V, Cb, 'g-', label='b')
        ax2.plot(V, Cc, 'b-', label='c')
        ax2.plot(V, Cd, 'y-', label='d')
        ax2.set_title('Concentration vs volume (PFR)')
        ax2.set_xlabel('Volume (L)')
        ax2.set_ylabel('Concentration (mol/L)')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    elif state == 'Gas':
        delta = (c / a + d / a - b / a - 1)
        epsilon = delta * ya
        Ca = c * (1 - x) / (1 + epsilon * x)
        Cb = c * (thetaB - (b / a) * x) / (1 + epsilon * x)
        Cc = c * (thetaC + (c / a) * x) / (1 + epsilon * x)
        Cd = c * (thetaD + (d / a) * x) / (1 + epsilon * x)
        ra1 = K*((c ** (a + b)) * ((1 - X) ** a) * ((thetaB - (b / a) * X) ** b))

        rA = -1 * ((c ** (a + b)) * ((1 - x) ** a) * ((thetaB - (b / a) * x) ** b) / (1 + epsilon * x) ** (a + b))

        for i in range(len(X)):
            def integrand(t):
                return (c*v) / ((c ** (a + b)) * ((1 - t) ** a) * ((thetaB - (b / a) * t) ** b))
            V[i] = quad(integrand, 0, X[i])[0]
        print(V)
        
        for i in range(len(x)):
            def integrand(t):
                return (c*v) / ((c ** (a + b)) * ((1 - t) ** a) * ((thetaB - (b / a) * t) ** b))
            V[i] = quad(integrand, 0, x[i])[0]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        ax1.plot(x, (-concA/ rA), 'b-')
        ax1.set_title('Levenspiel Plot (PFR)')
        ax1.set_xlabel('x')
        ax1.set_ylabel('-Ca0*V0/ rA (%)')

        ax2.plot(V, Ca, 'r-', label='a')
        ax2.plot(V, Cb, 'g-', label='b')
        ax2.plot(V, Cc, 'b-', label='c')
        ax2.plot(V, Cd, 'y-', label='d')
        ax2.set_title('Concentration vs volume (PFR)')
        ax2.set_xlabel('Volume (L)')
        ax2.set_ylabel('Concentration (mol/L)')
        ax2.legend()
        plt.show()
        plt.tight_layout()
       
 

def cstr_calculator(flowrate, K, concA, concB, concC, concD, a, b, c, d, X, state):
    
    x = np.linspace(0.00001, 0.9, 100)
    v = flowrate
    c = concA
    thetaB = concB/ concA
    thetaC = concC / concA
    thetaD = concD / concA

    if state == 'Liquid':
        
        Ca = c * (1 - x)
        Cb = c * (thetaB - (b / a) * x)
        Cc = c * (thetaC + (c / a) * x)
        Cd = c * (thetaD + (d / a) * x)
        #we consider all the reactions to be elementary here.

        rA = K*((c ** (a + b)) * ((1 - x) ** a) * ((thetaB - (b / a) * x) ** b))
        V = c*v*x/rA
        ra1 = K*((c ** (a + b)) * ((1 - X) ** a) * ((thetaB - (b / a) * X) ** b))
        V1 = c*v*X/ra1
        print(V1)
        
        # Plot results
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        ax1.plot(x, (c*v/ rA), 'b-')
        ax1.set_title('Levenspiel Plot (CSTR)')
        ax1.set_xlabel('x')
        ax1.set_ylabel('-Ca0*V0/ rA (%)')

        ax2.plot(V, Ca, 'r-', label='a')
        ax2.plot(V, Cb, 'g-', label='b')
        ax2.plot(V, Cc, 'b-', label='c')
        ax2.plot(V, Cd, 'y-', label='d')
        ax2.set_title('Concentration vs volume (CSTR)')
        ax2.set_xlabel('Volume (L)')
        ax2.set_ylabel('Concentration (mol/L)')
        ax2.legend()


        plt.savefig('/Users/satvikdeep/Documents/Reactor - Calculator /Templates/Conversion vs volume (CSTR).png')
        plt.savefig('/Users/satvikdeep/Documents/Reactor - Calculator /Templates/Levenspiel Plot (CSTR).png')
        
        plt.tight_layout()
        plt.show()

        
        
        
    


    elif state == 'Gas':
        delta = (c / a + d / a - b / a - 1)
        epsilon = delta * ya
        Ca = c * (1 - x) / (1 + epsilon * x)
        Cb = c * (thetaB - (b / a) * x) / (1 + epsilon * x)
        Cc = c * (thetaC + (c / a) * x) / (1 + epsilon * x)
        Cd = c * (thetaD + (d / a) * x) / (1 + epsilon * x)

        rA = -1 * ((c ** (a + b)) * ((1 - x) ** a) * ((thetaB - (b / a) * x) ** b) / (1 + epsilon * x) ** (a + b))
        V = (concA * x) / rA

        # Plot results
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        ax1.plot(x, (-concA*V0 / rA), 'b-')
        ax1.set_title('Levenspiel Plot (CSTR)')

# Test for a liquid-phase PFR
'''
# Test for a gas-phase CSTR
cstr_calculator(flowrate=200, K=10, concA=2, concB=2, concC=0, concD=0, a=1, b=1, c=1, d=1, X=0.7, state='Liquid')
'''
# Test for a gas-phase CSTR
cstr_calculator(flowrate=1, K=0.1, concA=1, concB=3.47, concC=0, concD=0, a=1, b=1, c=1, d=1, X=0.9, state='Liquid')

