import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from flask import Flask, request, render_template
from io import BytesIO
import base64



def pfr_calculator(flowrate, K, concA, concB, concC, concD, a, b, c, d, X, state):
    x = np.linspace(0.00001, 0.8, 10)
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
                return (-c*v)/ ((c ** (a + b)) * ((1 - t) ** a) * ((thetaB - (b / a) * t) ** b))
            V[i] = quad(integrand, 0, x[i])[0]
            print (V)
        
        print (Ca)
        print(Cb)
        print(Cc)
        
        
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
       
 

# Test for a liquid-phase PFR

pfr_calculator(flowrate=250, K=4.16, concA=2, concB=2, concC=2, concD=2, a=2, b=1, c=1, d=1, X=0.4, state='Liquid')
