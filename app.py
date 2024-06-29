import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from flask import Flask, request, render_template
from io import BytesIO
import base64
from threading import Thread


  # Use a backend that doesn't require a GUI

def run_matplotlib_code(func, *args, **kwargs):
    # Create a new thread for the main event loop
    matplotlib.pyplot.figure().canvas.manager.full_screen_window.wm_geometry("+%d+%d" % (100, 100))
    func(*args, **kwargs)
    plt.show(block=False)  # Show the plot non-blocking


def pfr_calculator(flowrate, K, concA, concB, concC, concD, a, b, c, d, X, state):
    x = np.linspace(0.00001, 0.9, 100)
    v = flowrate
    c = concA
    thetaB = concB/ concA
    thetaC = concC / concA
    thetaD = concD / concA
    
    if state == 'Liquid Phase':
        
        Ca = c * (1 - x)
        Cb = c * (thetaB - (b / a) * x)
        Cc = c * (thetaC + (c / a) * x)
        Cd = c * (thetaD + (d / a) * x)
        #we consider all the reactions to be elementary here.

        rA = K*((c ** (a + b)) * ((1 - x) ** a) * ((thetaB - (b / a) * x) ** b))

        def integrand(t):
            return K * (c * v) / ((c ** (a + b)) * ((1 - t) ** a) * ((thetaB - (b / a) * t) ** b))

        V = rA
        for i in range(len(x)):
            V[i] = quad(integrand, 0, x[i])[0]

        # Plot results
        # Plot results
        
        fig1, (ax1) = plt.subplots(1, 1, figsize=(4.5, 4.5))
        fig2, (ax2) = plt.subplots(1, 1, figsize=(4.5, 4.5))

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
        
        '''
        plt.savefig('/Users/satvikdeep/Documents/Reactor - Calculator /Templates/Conversion vs volume (CSTR).png')
        plt.savefig('/Users/satvikdeep/Documents/Reactor - Calculator /Templates/Levenspiel Plot (CSTR).png')
        plt.tight_layout()
        plt.show()
        '''

        return fig1, ax1, fig2, ax2

    elif state == 'Gas':
        delta = (c / a + d / a - b / a - 1)
        epsilon = delta * ya
        Ca = c * (1 - x) / (1 + epsilon * x)
        Cb = c * (thetaB - (b / a) * x) / (1 + epsilon * x)
        Cc = c * (thetaC + (c / a) * x) / (1 + epsilon * x)
        Cd = c * (thetaD + (d / a) * x) / (1 + epsilon * x)

        rA = -1 * ((c ** (a + b)) * ((1 - x) ** a) * ((thetaB - (b / a) * x) ** b) / (1 + epsilon * x) ** (a + b))

        V = rA
        for i in range(len(x)):
            def integrand(t):
                return flowRateA / ((c ** (a + b)) * ((1 - t) ** a) * ((thetaB - (b / a) * t) ** b))
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
        return plt.show()
        return plt.tight_layout()
       
 

def cstr_calculator(flowrate, K, concA, concB, concC, concD, a, b, c, d, X, state):
    
    x = np.linspace(0.00001, 0.9, 100)
    v = flowrate
    c = concA
    thetaB = concB/ concA
    thetaC = concC / concA
    thetaD = concD / concA
    
    if state == 'Liquid Phase':
        
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
        
        fig1, (ax1) = plt.subplots(1, 1, figsize=(4.5, 4.5))
        fig2, (ax2) = plt.subplots(1, 1, figsize=(4.5, 4.5))

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
        
        '''
        plt.savefig('/Users/satvikdeep/Documents/Reactor - Calculator /Templates/Conversion vs volume (CSTR).png')
        plt.savefig('/Users/satvikdeep/Documents/Reactor - Calculator /Templates/Levenspiel Plot (CSTR).png')
        plt.tight_layout()
        plt.show()
        '''

        return fig1, ax1, fig2, ax2
        


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

# Test for a gas-phase CSTR
cstr_calculator(flowrate=1, K=0.1, concA=1, concB=3.47, concC=0, concD=0, a=1, b=1, c=1, d=1, X=0.9, state='Liquid')
'''
def plot_to_base64(plot):
    # Save the plot to a buffer
    buffer = BytesIO()
    plot.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode the buffer contents as base64
    plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Close the buffer
    buffer.close()
    return plot_base64

app = Flask(__name__)



@app.route('/cstr', methods=['POST'])
def cstr():
    
    volumetric_flow_rate = float(request.form.get('volumetricFlowRate'))
    concentration_a = float(request.form.get('concentrationA'))
    concentration_b = float(request.form.get('concentrationB'))
    concentration_c = float(request.form.get('concentrationC'))
    concentration_d = float(request.form.get('concentrationD'))
    stoich_coeff_a = float(request.form.get('stoichCoeffA'))
    stoich_coeff_b = float(request.form.get('stoichCoeffB'))
    stoich_coeff_c = float(request.form.get('stoichCoeffC'))
    stoich_coeff_d = float(request.form.get('stoichCoeffD'))
    x = float(request.form.get('x'))
    rate_constant = float(request.form.get('rate_constant'))
    state = request.form.get('state')
    fig1, ax1, fig2, ax2 = cstr_calculator(volumetric_flow_rate, rate_constant, concentration_a, concentration_b, concentration_c, concentration_d, stoich_coeff_a, stoich_coeff_b, stoich_coeff_c, stoich_coeff_d, x, state)
    
    
    # Convert plots to base64 encoded strings
    c_vs_x_plot_encoded = plot_to_base64(fig1)
    levenspiel_plot_encoded = plot_to_base64(fig2)
    
    ra1 = rate_constant * ((concentration_a ** (stoich_coeff_a + stoich_coeff_b)) * ((1 - x) ** stoich_coeff_a) * ((concentration_b / concentration_a - (stoich_coeff_b / stoich_coeff_a) * x) ** stoich_coeff_b))
    V1 = concentration_a * volumetric_flow_rate * x / ra1


    return render_template('Page3.html', V1=V1, c_vs_x_plot=c_vs_x_plot_encoded, levenspiel_plot=levenspiel_plot_encoded)
    

@app.route('/pfr', methods=['POST'])
def prf():
    
    volumetric_flow_rate = float(request.form.get('volumetricFlowRate'))
    concentration_a = float(request.form.get('concentrationA'))
    concentration_b = float(request.form.get('concentrationB'))
    concentration_c = float(request.form.get('concentrationC'))
    concentration_d = float(request.form.get('concentrationD'))
    stoich_coeff_a = float(request.form.get('stoichCoeffA'))
    stoich_coeff_b = float(request.form.get('stoichCoeffB'))
    stoich_coeff_c = float(request.form.get('stoichCoeffC'))
    stoich_coeff_d = float(request.form.get('stoichCoeffD'))
    x = float(request.form.get('x'))
    rate_constant = float(request.form.get('rate_constant'))
    state = request.form.get('state')
    fig1, ax1, fig2, ax2 = pfr_calculator(volumetric_flow_rate, rate_constant, concentration_a, concentration_b, concentration_c, concentration_d, stoich_coeff_a, stoich_coeff_b, stoich_coeff_c, stoich_coeff_d, x, state)
    
    
    # Convert plots to base64 encoded strings
    c_vs_x_plot_encoded = plot_to_base64(fig1)
    levenspiel_plot_encoded = plot_to_base64(fig2)

    rA = rate_constant*((concentration_a ** (stoich_coeff_a + stoich_coeff_b)) *  ((1 - x) ** stoich_coeff_a) * ((concentration_b / concentration_a - (stoich_coeff_b / stoich_coeff_a) * x) ** stoich_coeff_b))


    def integrand(t):
        return (concentration_a * volumetric_flow_rate) / rate_constant*((concentration_a ** (stoich_coeff_a + stoich_coeff_b)) *  ((1 - t) ** stoich_coeff_a) * ((concentration_b / concentration_a - (stoich_coeff_b / stoich_coeff_a) * t) ** stoich_coeff_b))

    V = rA
    for i in range(len(x)):
        V[i] = quad(integrand, 0, x[i])[0]

    print (V)
    return render_template('Page3.html', V1=V, c_vs_x_plot=c_vs_x_plot_encoded, levenspiel_plot=levenspiel_plot_encoded)
   







