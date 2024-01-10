import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time

from sympy.functions.special.gamma_functions import uppergamma
from scipy.special import lambertw
from scipy.optimize import minimize

import warnings
warnings.filterwarnings('ignore')

# ----------------------------- Analytic PDFs -----------------------------
def power_pdf(x:np.array, alpha:float=1.5, x_min:float=1):
    """
    Potential Probability Distribution Function (PDF) for the 1D case.

    Parameters:
        x (float or array-like of floats): The input value(s).
        alpha (float): The power value.
        x_min (float): The starting point.

    Returns:
        The PDF evaluated at the input values.
    """

    x = np.array(x)
    if x.ndim > 1:
        raise ValueError("x should be a float or a 1D array-like of floats.")

    fx = ((alpha - 1) / x_min)*(x/x_min)**(-alpha)

    return fx

def damped_power_pdf(x:np.array, alpha:float=1.5, x_c:float=100, x_min:float=1):
    """
    Damped Potential Probability Distribution Function (PDF) for the 1D case.

    Parameters:
        x (float or array-like of floats): The input value(s).
        alpha (float): The power value.
        x_min (float): The starting point.
        x_c (float): The damping coefficient.

    Returns:
        The PDF evaluated at the input values.
    """
    x = np.array(x)
    if x.ndim > 1:
        raise ValueError("x should be a float or a 1D array-like of floats.")

    C1 = (x_c ** (alpha - 1)) / np.float64(uppergamma(1-alpha, x_min/x_c).evalf())
    fx = C1 * np.exp(-x/x_c) * (x**(-alpha))

    return fx

def dslope_power_pdf(x:np.array, alpha1:float=4, alpha2:float=1.5, x_c:float=100, beta:float=0.5, x_min:float=1):
    """
    Damped Potential with Double Slope Probability Distribution Function (PDF) for the 1D case.

    Parameters:
        n_samples (int): The number of datapoints desired.
        alpha1 (float): The pure power value.
        alpha2 (float): The cutoff power value.
        x_min (float): The starting point.
        x_c (float): The damping coefficient.

    Returns:
        The PDF evaluated at the input values.
    """
    C1 = beta/(1-beta) * np.float64(((alpha1-1)*uppergamma(1-alpha2, x_min/x_c)) / ((x_min**(1-alpha1))*(x_c**(1-alpha2))))
    tau = np.float64((alpha2 - alpha1)*x_c*lambertw(C1**(-1/(alpha2-alpha1))/((alpha2-alpha1)*x_c)))

    pob1_part = beta * ((alpha1 - 1) / x_min) * (x/x_min)**(-alpha1)
    pob2_part = (1 - beta) * ((x_c ** (alpha2 - 1)) / np.float64(uppergamma(1-alpha2, x_min/x_c))) * ((x/x_min)**(-alpha2)) * np.exp(-x/x_c)
    px = pob1_part + pob2_part
    return px, tau

# ----------------------------- Analytic CDF -----------------------------
def power_ccdf(x:np.array, alpha:float=1.5, x_min:float=1):
    """
    Accumulated Potential Probability Density Function (CDF) for the 1D case.
    Parameters:
        x (float or array-like of floats): The input value(s).
        alpha (float): The power value.
        x_min (float): The starting point.

    Returns:
        The CDF evaluated at the input values.
    """

    x = np.array(x)
    if x.ndim > 1:
        raise ValueError("x should be a float or a 1D array-like of floats.")

    cfx = (x/x_min)**(1-alpha)

    return cfx

def damped_power_ccdf(x:np.array, alpha:float=1.5, x_c:float=100, x_min:float=1):
    """
    Accumulated Potential Probability Density Function (CDF) for the 1D case.
    Parameters:
        x (float or array-like of floats): The input value(s).
        alpha (float): The power value.
        x_min (float): The starting point.

    Returns:
        The CDF evaluated at the input values.
    """

    x = np.array(x)
    if x.ndim > 1:
        raise ValueError("x should be a float or a 1D array-like of floats.")

    norm = uppergamma(1-alpha, x_min/x_c)
    vuppergamma = np.vectorize(uppergamma)

    cfx = np.float64(vuppergamma(1-alpha, x/x_c)) / np.float64(norm)
    return cfx

def dslope_power_ccdf(x:np.array, alpha1:float=4, alpha2:float=1.5, x_c:float=100, x_min:float=1):
    pass

# ----------------------------- PDF's Derivates ----------------------------- WIP - May not be needed

# ----------------------------- Generation of the PDFs -----------------------------
def analytic_power(n_samples:int, fx:np.array=None, alpha:float=2, x_min:float=1):
    """
    Potential Probability Distribution Function (PDF) for the 1D case.

    Parameters:
        n_samples (int): The number of samples to generate.
        alpha (float): The power value.
        x_min (float): The starting point.

    Returns:
        The PDF evaluated at the input values.
    """
    
    x = np.random.uniform(size=n_samples)
    
    if fx is None:
        x = x_min*((1-x)**(1/(1-alpha)))
    
    # cx = 1 - (x/x_min) ** (1-alpha) 
    return x

def accept_reject_damped(n_samples:int, alpha:float=1.5, x_c:float=100, x_min:float=1):
    """
    Generate random samples from an unbounded uniform distribution using the acceptance-rejection method.

    Parameters:
        n_samples (int): The number of samples to generate.
        a (float): The lower bound of the bounded uniform distribution.
        b (float): The upper bound of the bounded uniform distribution.
        top_level (float): The maximum value of the potential PDF.

    Returns:
        samples (array-like of floats): The generated samples.
    """
    x = []
    efficiency = 0
    i = 0

    while len(x) < n_samples:
        i+=1
        Y = analytic_power(n_samples - len(x), alpha=alpha, x_min=x_min)
        u = np.random.uniform(size=n_samples - len(x))
        x.extend(Y[np.where(u < np.exp((x_min-Y)/x_c))])
        efficiency += (n_samples - len(x))
        print(f'{n_samples - len(x)} left and iter number {i}', end='\r')
    efficiency = 100 * n_samples / efficiency
    print(f'Generated {n_samples} in {i} iterations with {np.round(efficiency, 2)}% efficiency')

    x = np.array(x)
    return x

def double_potential_generation(n_samples:int, alpha1:float=4, alpha2:float=1.5, x_c:float=100, beta:float=0.5, x_min:float=1):
    """
    Damped Potential with Double Slope Probability Distribution Function (PDF) for the 1D case.

    Parameters:
        n_samples (int): The number of datapoints desired.
        alpha1 (float): The pure power value.
        alpha2 (float): The cutoff power value.
        x_min (float): The starting point.
        x_c (float): The damping coefficient.

    Returns:
        The PDF evaluated at the input values.
    """

    x = np.random.uniform(size=n_samples)
    pop_a = len(x[np.where(x < beta)])

    # Population A: Pure Power Law
    x1 = analytic_power(pop_a, alpha=alpha1, x_min=x_min)
    
    # Population B: Cutoff Power Law
    x2 = accept_reject_damped(n_samples - pop_a, alpha=alpha2, x_c=x_c, x_min=x_min)

    x = np.concatenate([x1, x2], axis=0)
    """
    input prob de 1
    u = [1, 1, 0, 0, 1, 0, ...] -> 1 pob A, 0 pob B
    u2 = uniforme

    u2a = [mask 1] -> pure power -> (new x) -> (u2a, fx)
    u2b = [mask 0] -> cutoff  ...           -> (u2b, fx')  -> Histograma

    pdf(x) = f(x)dx -> ln(pdf) = ln(f) + ln(dx) Revisar
    """
    return x

# ----------------------------- Likelihood Functions -----------------------------
def power_law_max_likelihood(x:np.array, x_min:float=1):
    return 1 + len(x) / np.sum(np.log(x/x_min))

def power_law_likelihood(params:tuple, x:np.array, x_min:float=1):
    alpha = params[0]
    return -len(x) * np.log((alpha - 1)/x_min) + alpha * np.sum(np.log(x/x_min))

def cutoff_law_likelihood(params:list, x:np.array, x_min:float=1):
    alpha, x_c = params
    return len(x) * np.log(np.float64(uppergamma(1-alpha, x_min/x_c)) / (x_c ** (alpha - 1))) + alpha * np.sum(np.log(x)) + np.sum(x) / x_c

def dslope_law_likelihood(params:list, x:np.array, beta:float=0.5, x_min:float=1):
    alpha1, alpha2, x_c = params
    # print(alpha1, alpha2, x_c)
    with open('convergence.csv', 'a') as f:
        f.write(f'{alpha1},{alpha2},{x_c},{uppergamma(1-alpha2, x_min/x_c).evalf()}\n')
    pob1_part = beta * ((alpha1 - 1) / x_min) * (x/x_min)**(-alpha1)
    pob2_part = (1 - beta) * ((x_c ** (alpha2 - 1)) / np.float64(uppergamma(1-alpha2, x_min/x_c))) * ((x/x_min)**(-alpha2)) * np.exp(-x/x_c)
    return -np.sum(np.log(pob1_part + pob2_part))

def dslope_noprob_law_likelihood(params:list, x:np.array, x_min:float=1):
    alpha1, alpha2, x_c, beta = params
    # print(alpha1, alpha2, x_c, beta)
    with open('convergence_noprob.csv', 'a') as f:
        f.write(f'{alpha1},{alpha2},{x_c},{beta},{uppergamma(1-alpha2, x_min/x_c).evalf()}\n')
    pob1_part = beta * ((alpha1 - 1) / x_min) * (x/x_min)**(-alpha1)
    pob2_part = (1 - beta) * ((x_c ** (alpha2 - 1)) / np.float64(uppergamma(1-alpha2, x_min/x_c))) * ((x/x_min)**(-alpha2)) * np.exp(-x/x_c)
    return -np.sum(np.log(pob1_part + pob2_part))

# ----------------------------- Minimization Algorythms -----------------------------
def minimize_by_scipy(x:np.array, likelihood, guess:list, x_min:float=1, method:str='Nelder-Mead', bounds:tuple=None):
    # return  minimize(likelihood, guess, args=(x,), method=method) # options={'disp':True}
    return  minimize(likelihood, guess, args=(x,), method=method, bounds=bounds) # options={'disp':True}

def minimize_by_gradient(guess:list, x:np.array, fx:np.array, pdf, rho:float=0.01, c:float=0.00001, max_iter:float=10000): # WIP
    new_fx = pdf(x, 1, *guess)
    grad_f = np.gradient(new_fx, x)
    mse = np.mean((fx - new_fx)**2)
    print('\n')

    for i, _ in enumerate(range(max_iter)):
        guess = guess - rho * np.array([np.sum(grad_f * (fx - new_fx), axis=0) / len(x), np.sum((fx - new_fx) * x**2 / guess[1], axis=0) / len(x)])
        new_fx = pdf(x, 1, *guess)
        mse_new = np.mean((fx - new_fx)**2)

        print(f'Iter number {i} with error {round(mse_new, 6)}', end='\r')
        if abs(mse_new - mse) < c:
            print('The Method Converged!                   ')
            break
        mse = mse_new
    return guess, mse_new

# ----------------------------- Hessian Computation ----------------------------- # NEEDS OPTIMIZATION
def power_hessian(x:np.array, alpha:float=1.5, x_min:float=1):
    return len(x)/(alpha-1)**2

def damped_power_hessian(x:np.array, alpha:float=1.5, x_c:float=100, x_min:float=1):
    vuppergamma = np.vectorize(lambda s,x: np.float64(uppergamma(s, x)))
    gamma_values = vuppergamma(1-alpha, x/x_c)
    dgamma, d2gamma = d_uppergamma(x, alpha, x_c, x_min=x_min)
    hess =  np.sum(np.array([[d2gamma[0, 0, :]/gamma_values - (dgamma[0, :]/gamma_values)**2, np.log(x/x_min)/x_c + d2gamma[0, 1, :]/gamma_values - dgamma[0]*dgamma[1]/gamma_values**2],
                            [d2gamma[1, 0, :]/gamma_values - dgamma[0]*dgamma[1]/gamma_values**2 - 1/x_c, (alpha - 1)/x_c**2 + 2*x/x_c**3 + d2gamma[1, 1, :]/gamma_values - (dgamma[1, :]/gamma_values)**2]]), axis=2)
    return hess

def dslope_power_hessian(x:np.array, alpha1:float=4, alpha2:float=1.5, x_c:float=100, beta:float=0.5, x_min:float=1):
    vuppergamma = np.vectorize(lambda s,x: np.float64(uppergamma(s, x)))
    dgamma, d2gamma = d_uppergamma(x, alpha2, x_c, x_min=x_min)
    gamma_values = vuppergamma(1-alpha2, x/x_c)
    px, tau = dslope_power_pdf(x, alpha1=alpha1, alpha2=alpha2, x_c=x_c, beta=beta, x_min=x_min)
    hess1 = beta * np.sum(power_pdf(x, alpha=alpha1, x_min=x_min)/((alpha1-1)**2) / px)
    hess2 = (1 - beta) * np.sum(damped_power_pdf(x, alpha=alpha2, x_c=x_c, x_min=x_min) * np.array([[d2gamma[0, 0, :]/gamma_values - (dgamma[0, :]/gamma_values)**2, np.log(x/x_min)/x_c + d2gamma[0, 1, :]/gamma_values - dgamma[0]*dgamma[1]/gamma_values**2],
                                                                                                    [d2gamma[1, 0, :]/gamma_values - dgamma[0]*dgamma[1]/gamma_values**2 - 1/x_c, (alpha2 - 1)/x_c**2 + 2*x/x_c**3 + d2gamma[1, 1, :]/gamma_values - (dgamma[1, :]/gamma_values)**2]])/px, axis=2) 
    hess = np.eye(3)
    hess[0, 0] = hess1
    hess[1:, 1:] = hess2
    return hess

def dslope_power_no_prob_hessian(x:np.array, alpha1:float=4, alpha2:float=1.5, x_c:float=100, beta:float=0.5, x_min:float=1):
    vuppergamma = np.vectorize(lambda s,x: np.float64(uppergamma(s, x)))
    dgamma, d2gamma = d_uppergamma(x, alpha2, x_c, x_min=x_min)
    gamma_values = vuppergamma(1-alpha2, x/x_c)
    px, tau = dslope_power_pdf(x, alpha1=alpha1, alpha2=alpha2, x_c=x_c, beta=beta, x_min=x_min)
    pure_power = power_pdf(x, alpha=alpha1, x_min=x_min)
    damped_power = damped_power_pdf(x, alpha=alpha2, x_c=x_c, x_min=x_min)
    hess = np.eye(4)

    # First Value
    hess1 = beta * np.sum(pure_power /(px*(alpha1-1)**2))
    hess[0, 0] = hess1
    
    # Center Square
    hess2 = (1 - beta) * np.sum(damped_power * np.array([[d2gamma[0, 0, :]/gamma_values - (dgamma[0, :]/gamma_values)**2, np.log(x/x_min)/x_c + d2gamma[0, 1, :]/gamma_values - dgamma[0]*dgamma[1]/gamma_values**2],
                                                         [d2gamma[1, 0, :]/gamma_values - dgamma[0]*dgamma[1]/gamma_values**2 - 1/x_c, (alpha2 - 1)/x_c**2 + 2*x/x_c**3 + d2gamma[1, 1, :]/gamma_values - (dgamma[1, :]/gamma_values)**2]])/px, axis=2) 
    hess[1:3, 1:3] = hess2

    # Last Row
    hess[-1, :-1] = np.sum(np.array([beta*pure_power*(1/(alpha1-1)-np.log(x/x_min)), (1-beta)*damped_power*(np.log(x/(x_c*x_min))+dgamma[0]/gamma_values), (1-beta)*damped_power*((1-alpha2)/x_c-x/x_c**2+dgamma[1]/gamma_values)]) / px, axis=1)
    
    # Last Column
    hess[:-1, -1] = np.sum(np.array([pure_power*(1/(alpha1-1)-np.log(x/x_min)), damped_power*(np.log(x/(x_c*x_min))+dgamma[0]/gamma_values), damped_power*((1-alpha2)/x_c-x/x_c**2+dgamma[1]/gamma_values)]) / px, axis=1)
    return hess

# ----------------------------- Data Plots -----------------------------
def plot_data(x:np.array, n_bins:int, x_estimated:np.array, px_bootstrap:np.array, px:np.array, errors:np.array, title=''):
    sns.set_style('darkgrid')

    # Preparation of the errorbars
    non_boostrap_errors, bootstrap_errors = errors[0, :], errors[1, :]
    bootstrap_errors_sup, bootstrap_errors_inf = bootstrap_errors[0, :], bootstrap_errors[1, :]
    non_bootstrap_errors_sup, non_bootstrap_errors_inf = non_boostrap_errors[0, :], non_boostrap_errors[1, :]
    
    # Preparation of the datapoints
    hist, bin_edges = np.histogram(np.log10(x), bins=n_bins)
    bin_width = [np.power(10, bin_edges[i + 1]) - np.power(10, bin_edges[i]) for i in range(len(bin_edges) - 1)]
    t_hist = hist.sum()
    hist = hist/bin_width / t_hist
    x_hist = [np.log10(np.min(x)) + np.log10(np.max(x) - np.min(x))*(i) / n_bins for i in range(len(hist))]
    y_base = np.ones(len(x_hist))/bin_width / t_hist

    plt.figure()
    # First Subplot: Non-Bootstrapped Data
    sns.scatterplot(x=np.power(10, x_hist), y=hist)
    sns.lineplot(x=np.power(10, x_hist), y=y_base, color='black', label='Unitary Bins', linestyle='--', alpha=0.5)

    sns.lineplot(x=x_estimated, y=px, color='blue', label='Non-Bootstrapped Data', alpha=0.5)
    sns.lineplot(x=x_estimated, y=px_bootstrap, color='green', label='Bootstrapped Data', alpha=0.5)

    sns.lineplot(x=x_estimated, y=bootstrap_errors_sup, color='red', label=None, alpha=0.2, linestyle='--')
    sns.lineplot(x=x_estimated, y=bootstrap_errors_inf, color='red', label=None, alpha=0.2, linestyle='--')
    
    plt.title(title)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel("PDF(x)")
    plt.xlabel("x")

    plt.legend()
    plt.show()

# ----------------------------- Storing And Reading Pickled Data -----------------------------
def save_data(x, y, name, label):
    combined_arrays = {'x': x, 'y': y}

    with open(f"data/{name}/{label:{4}}.txt".replace(" ", "0"), 'wb') as f:
        pickle.dump(combined_arrays, f)

def load_data(name, label):
    with open(f"data/{name}/{label:{4}}.txt".replace(" ", "0"), 'rb') as f:
        loaded_combined_arrays = pickle.load(f)
    
    x_data = loaded_combined_arrays['x']
    y_data = loaded_combined_arrays['y']
    return x_data, y_data

# ----------------------------- Main PDFs Generation Workflow -----------------------------
def generate_power_law_sample(x:np.array=None, n_samples:int=None, alpha:float=1.5, x_min:float=1, show_data:bool=False, n_bins:int=100, bootstrap:int=None):
    """
    Generate a sample from a power law distribution
    """

    # Analysis of the power PDF (Generation + Likelihood)
    if (x is None) and (n_samples is not None):
        # Generation
        print('Generating Samples...')
        x = analytic_power(n_samples, alpha=alpha, x_min=x_min)
        print('Samples Done!')

    # Maximum Likelihood Estimation
    if bootstrap is not None:
        alpha_pred = np.array([])
        for _ in range(bootstrap):
            xb = bootstrap_sample(x)
            alpha_pred = np.append(alpha_pred, power_law_max_likelihood(xb))
        # Alpha
        alpha_std_bootstrap = bootstrap_std(alpha_pred, bootstrap)
        alpha_pred_bootstrap = np.sum(alpha_pred) / bootstrap
    # else:
    #     alpha_pred = power_law_likelihood(x)
    
    alpha_pred = power_law_max_likelihood(x)
    """
    alpha0 = 2.5
    result = minimize_by_scipy(x, power_law_likelihood, [alpha0], method='SLSQP', bounds=[(1, 6)])
    alpha_pred = result.x[0]
    """

    cp1 = time.perf_counter()
    hess = power_hessian(x, alpha_pred)
    cp2 = time.perf_counter()
    print(f'Time ellapsed to Compute the Hessian: {round(1000 * (cp2-cp1), 3)}ms')
    alpha_std = np.sqrt(1/hess)
    akaike = akaike_information(x, power_law_likelihood, [alpha_pred], x_min=x_min)
    bayes = bayesian_information(x, power_law_likelihood, [alpha_pred], x_min=x_min)

    x_estimated = np.linspace(min(x), max(x), len(x))
    px = power_pdf(x_estimated, alpha=alpha_pred, x_min=x_min)
    px_bootstrap = power_pdf(x_estimated, alpha=alpha_pred_bootstrap, x_min=x_min)


    errors_non_bootstrap = np.array([power_pdf(x_estimated, alpha=alpha_pred + (-1)**(i//1) * alpha_std, x_min=x_min) for i in range(2)])
    errors_non_bootstrap = np.array([np.max(errors_non_bootstrap, axis=0), np.min(errors_non_bootstrap, axis=0)])

    errors_bootstrap = np.array([power_pdf(x_estimated, alpha=alpha_pred_bootstrap + (-1)**(i//1) * alpha_std_bootstrap, x_min=x_min) for i in range(2)])
    errors_bootstrap = np.array([np.max(errors_bootstrap, axis=0), np.min(errors_bootstrap, axis=0)])

    errors = np.array([errors_non_bootstrap, errors_bootstrap])
    # Plot
    if show_data:
        print(f'Akaike Information Criteria: {round(akaike, 4)}\tBayesian Information Criteria: {round(bayes, 4)}')
        print(f'\nNon-Bootstrapped Samples:\n\tSlope Predicted: {alpha_pred} +- {alpha_std}')
        print(f'\nBootstrapped Samples:\n\tSlope Predicted: {alpha_pred_bootstrap} +- {alpha_std_bootstrap}')
        plot_data(x, n_bins, x_estimated, px_bootstrap, px, errors, title='Pure Power Law')
    
    return alpha_pred

def generate_cutoff_law_sample(x:np.array=None, n_samples:int=None, alpha:float=1.5, x_c:float=100, x_min:float=1, show_data:bool=False, n_bins:int=100, bootstrap:int=False):
    # Analysis of the power with cutoff PDF (Generation + Likelihood)
    if (x is None) and (n_samples is not None):
        # Generation
        print('Generating Samples...')
        x = accept_reject_damped(n_samples, alpha=alpha, x_c=x_c, x_min=x_min)

    # Maximum Likelihood Estimation
    alpha0, x_c0 = alpha + np.random.uniform(-1, 1, 1), x_c + np.random.uniform(-10, 10, 1)
    if bootstrap is not None:
        alpha_pred = np.array([])
        x_c_pred = np.array([])
        for _ in range(bootstrap):
            xb = bootstrap_sample(x)
            result = minimize_by_scipy(xb, cutoff_law_likelihood, (alpha0, x_c0), method='SLSQP', bounds=((1.2, 6), (0.00001, None)))
            alpha_pred = np.append(alpha_pred, result.x[0])
            x_c_pred = np.append(x_c_pred, result.x[1])
        # Alpha
        alpha_std_bootstrap = bootstrap_std(alpha_pred, bootstrap)
        alpha_pred_bootstrap = np.sum(alpha_pred)/bootstrap
        # X_c
        x_c_std_bootstrap = bootstrap_std(x_c_pred, bootstrap)
        x_c_pred_bootstrap = np.sum(x_c_pred)/bootstrap
    # else:
    #     result = minimize_by_scipy(x, cutoff_law_likelihood, (alpha0, x_c0))
    #     alpha_pred, x_c_pred = result.x[0], result.x[1]
    result = minimize_by_scipy(x, cutoff_law_likelihood, (alpha0, x_c0), method='SLSQP', bounds=((1.2, 6), (0.00001, None)))
    alpha_pred, x_c_pred = result.x[0], result.x[1]
    cp1 = time.perf_counter()
    hess = damped_power_hessian(x, alpha_pred, x_c_pred)
    cp2 = time.perf_counter()
    print(f'Time ellapsed to Compute the Hessian: {round(1000 * (cp2-cp1), 3)}ms')
    alpha_std, x_c_std = np.diagonal(np.sqrt(abs(np.linalg.inv(hess))))
    akaike = akaike_information(x, cutoff_law_likelihood, (alpha_pred, x_c_pred), x_min=x_min)
    bayes = bayesian_information(x, cutoff_law_likelihood, (alpha_pred, x_c_pred), x_min=x_min)

    x_estimated = np.linspace(min(x), max(x), len(x))
    px = damped_power_pdf(x_estimated, alpha=alpha_pred, x_c=x_c_pred, x_min=x_min)
    px_bootstrap = damped_power_pdf(x_estimated, alpha=alpha_pred_bootstrap, x_c=x_c_pred_bootstrap, x_min=x_min)

    errors_non_bootstrap = []
    errors_bootstrap = []
    for i in range(4):
        alpha2test = max(0.5, alpha_pred + (-1)**(i//1)*alpha_std)
        xc2test = max(1, x_c_pred + (-1)**(i//2)*x_c_std)
        errors_non_bootstrap.append(damped_power_pdf(x_estimated, alpha=alpha2test, x_c=xc2test, x_min=x_min))

        alpha2test_b = max(0.5, alpha_pred_bootstrap + (-1)**(i//1)*alpha_std_bootstrap)
        xc2test_b = max(1, x_c_pred_bootstrap + (-1)**(i//2)*x_c_std_bootstrap)
        errors_bootstrap.append(damped_power_pdf(x_estimated, alpha=alpha2test_b, x_c=xc2test_b, x_min=x_min))

    errors_non_bootstrap = np.array([np.max(np.array(errors_non_bootstrap), axis=0), np.min(np.array(errors_non_bootstrap), axis=0)])
    errors_bootstrap = np.array([np.max(np.array(errors_bootstrap), axis=0), np.min(np.array(errors_bootstrap), axis=0)])

    errors = np.array([errors_non_bootstrap, errors_bootstrap])
    # Plot
    if show_data:
        print(f'Akaike Information Criteria: {round(akaike, 4)}\tBayesian Information Criteria: {round(bayes, 4)}')
        print(f'\nNon-Bootstrapped Samples:\n\tSlope Predicted: {alpha_pred} +- {alpha_std}\n\tDamping Factor Predicted: {x_c_pred} +- {x_c_std}')
        print(f'\nBootstrapped Samples:\n\tSlope Predicted: {alpha_pred_bootstrap} +- {alpha_std_bootstrap}\n\tDamping Factor Predicted: {x_c_pred_bootstrap} +- {x_c_std_bootstrap}')
        plot_data(x, n_bins, x_estimated, px_bootstrap, px, errors, title='Power Law with Exponential Cutoff')

    return alpha_pred, x_c_pred

def generate_dslope_law_sample(x:np.array=None, n_samples:int=None, alpha1:float=4, alpha2:float=1.5, x_c:float=100, beta:float=0.5, x_min:float=1, show_data:bool=False, n_bins:int=100, bootstrap:int=False):
    # Analysis of the double slope power PDF (Generation + Likelihood)
    if (x is None) and (n_samples is not None):
        # Generation
        print('Generating Samples...')
        x = double_potential_generation(n_samples, alpha1=alpha1, alpha2=alpha2, x_c=x_c, beta=beta, x_min=x_min)

    # Maximum Likelihood Estimation
    alpha1_0, alpha2_0, x_c0 = alpha1 + np.random.uniform(-1, 1, 1), alpha2 + np.random.uniform(-1, 1, 1), x_c + np.random.uniform(-10, 10, 1)
    if bootstrap is not None:
        alpha1_pred = np.array([])
        alpha2_pred = np.array([])
        x_c_pred = np.array([])
        for _ in range(bootstrap):
            xb = bootstrap_sample(x)
            result = minimize_by_scipy(xb, dslope_law_likelihood, (alpha1_0, alpha2_0, x_c0), method='SLSQP', bounds=((1.2, 6), (0.5, 2), (0.00001, None))) # method='SLSQP'
            alpha1_pred = np.append(alpha1_pred, result.x[0])
            alpha2_pred = np.append(alpha2_pred, result.x[1])
            x_c_pred = np.append(x_c_pred, result.x[2])
        
        # Alpha1
        alpha1_std_bootstrap = bootstrap_std(alpha1_pred, bootstrap)
        alpha1_pred_bootstrap = np.sum(alpha1_pred)/bootstrap
        # Alpha2
        alpha2_std_bootstrap = bootstrap_std(alpha2_pred, bootstrap)
        alpha2_pred_bootstrap = np.sum(alpha2_pred)/bootstrap
        # X_c
        x_c_std_bootstrap = bootstrap_std(x_c_pred, bootstrap)
        x_c_pred_bootstrap = np.sum(x_c_pred)/bootstrap
    else:
        result = minimize_by_scipy(x, dslope_law_likelihood, (alpha1_0, alpha2_0, x_c0))
        alpha1_pred, alpha2_pred, x_c_pred = result.x[0], result.x[1], result.x[2]

    result = minimize_by_scipy(x, dslope_law_likelihood, (alpha1_0, alpha2_0, x_c0), bounds=((1.2, 6), (0.5, 2), (0.00001, None)))
    alpha1_pred, alpha2_pred, x_c_pred = result.x[0], result.x[1], result.x[2]
    cp1 = time.perf_counter()
    hess = dslope_power_hessian(x, alpha1_pred, alpha2_pred, x_c_pred)
    cp2 = time.perf_counter()
    print(f'Time ellapsed to Compute the Hessian: {round(1000 * (cp2-cp1), 3)}ms')
    alpha1_std, alpha2_std, x_c_std = np.diagonal(np.sqrt(abs(np.linalg.inv(hess))))
    akaike = akaike_information(x, dslope_law_likelihood, (alpha1_pred, alpha2_pred, x_c_pred), x_min=x_min)
    bayes = bayesian_information(x, dslope_law_likelihood, (alpha1_pred, alpha2_pred, x_c_pred), x_min=x_min)

    x_estimated = np.linspace(min(x), max(x), len(x))
    px_bootstrap, tau_b = dslope_power_pdf(x_estimated, alpha1=alpha1_pred_bootstrap, alpha2=alpha2_pred_bootstrap, x_c=x_c_pred_bootstrap, beta=beta, x_min=x_min)
    px, tau = dslope_power_pdf(x_estimated, alpha1=alpha1_pred, alpha2=alpha2_pred, x_c=x_c_pred, beta=beta, x_min=x_min)

    errors_non_bootstrap = []
    errors_bootstrap = []
    for i in range(8):
        alpha12test = max(2, alpha1_pred + (-1)**(i//1)*alpha1_std)
        alpha22test = min(alpha12test, max(0.5, alpha2_pred + (-1)**(i//2)*alpha2_std))
        xc2test = max(1, x_c_pred + (-1)**(i//4)*x_c_std)
        errors_non_bootstrap.append(dslope_power_pdf(x_estimated, alpha1=alpha12test, alpha2=alpha22test, x_c=xc2test, beta=beta, x_min=x_min)[0])

        alpha12test_b = max(2, alpha1_pred_bootstrap + (-1)**(i//1)*alpha1_std_bootstrap)
        alpha22test_b = min(alpha12test_b, max(0.5, alpha2_pred_bootstrap + (-1)**(i//2)*alpha2_std_bootstrap))
        xc2test_b = max(1, x_c_pred_bootstrap + (-1)**(i//4)*x_c_std_bootstrap)
        # print(alpha12test_b, alpha22test_b, xc2test_b)
        errors_bootstrap.append(dslope_power_pdf(x_estimated, alpha1=alpha12test_b, alpha2=alpha22test_b, x_c=xc2test_b, beta=beta, x_min=x_min)[0])

    errors_non_bootstrap = np.array([np.max(np.array(errors_non_bootstrap), axis=0), np.min(np.array(errors_non_bootstrap), axis=0)])
    errors_bootstrap = np.array([np.max(np.array(errors_bootstrap), axis=0), np.min(np.array(errors_bootstrap), axis=0)])

    errors = np.array([errors_non_bootstrap, errors_bootstrap])
    # Plot
    if show_data:
        print(f'Akaike Information Criteria: {round(akaike, 4)}\tBayesian Information Criteria: {round(bayes, 4)}')
        print(f'\nNon-Bootstrapped Samples:\n\tSlope 1 Predicted: {alpha1_pred} +- {alpha1_std}\n\tSlope 2 Predicted: {alpha2_pred} +- {alpha2_std}\n\tDamping Factor Predicted: {x_c_pred} +- {x_c_std}\n\tCrossing Point Predicted: {tau}')
        print(f'\nBootstrapped Samples:\n\tSlope 1 Predicted: {alpha1_pred_bootstrap} +- {alpha1_std_bootstrap}\n\tSlope 2 Predicted: {alpha2_pred_bootstrap} +- {alpha2_std_bootstrap}\n\tDamping Factor Predicted: {x_c_pred_bootstrap} +- {x_c_std_bootstrap}\n\tCrossing Point Predicted: {tau_b}')
        plot_data(x, n_bins, x_estimated, px_bootstrap, px, errors, title='Double Power Law with Exponential Cutoff')

    return alpha1_pred, alpha2_pred, x_c_pred

def generate_dslope_noprob_law_sample(x:np.array=None, n_samples:int=None, alpha1:float=4, alpha2:float=1.5, x_c:float=100, beta:float=0.5, x_min:float=1, show_data:bool=False, n_bins:int=100, bootstrap:int=False):

    # Analysis of the double slope power PDF (Generation + Likelihood)
    if (x is None) and (n_samples is not None):
        # Generation
        print('Generating Samples...')
        x = double_potential_generation(n_samples, alpha1=alpha1, alpha2=alpha2, x_c=x_c, beta=beta, x_min=x_min)

    # Maximum Likelihood Estimation
    alpha1_0, alpha2_0, x_c0, beta0 = alpha1 + np.random.uniform(-1, 1, 1), alpha2 + np.random.uniform(-1, 1, 1), x_c + np.random.uniform(-10, 10, 1), beta + np.random.uniform(-0.1, 0.1, 1)
    if bootstrap is not None:
        alpha1_pred = np.array([])
        alpha2_pred = np.array([])
        x_c_pred = np.array([])
        beta_pred = np.array([])
        for _ in range(bootstrap):
            xb = bootstrap_sample(x)
            result = minimize_by_scipy(xb, dslope_noprob_law_likelihood, (alpha1_0, alpha2_0, x_c0, beta0), method='SLSQP', bounds=((2, 6), (0.5, 2), (0.00001, None), (0.000001, 0.999999)))
            alpha1_pred = np.append(alpha1_pred, result.x[0])
            alpha2_pred = np.append(alpha2_pred,result.x[1])
            x_c_pred = np.append(x_c_pred,result.x[2])
            beta_pred = np.append(beta_pred,result.x[3])

        # Alpha1
        alpha1_std_bootstrap = bootstrap_std(alpha1_pred, bootstrap)
        alpha1_pred_bootstrap = np.sum(alpha1_pred)/bootstrap
        # Alpha2
        alpha2_std_bootstrap = bootstrap_std(alpha2_pred, bootstrap)
        alpha2_pred_bootstrap = np.sum(alpha2_pred)/bootstrap
        # X_c
        x_c_std_bootstrap = bootstrap_std(x_c_pred, bootstrap)
        x_c_pred_bootstrap = np.sum(x_c_pred)/bootstrap
        # Beta
        beta_std_bootstrap = bootstrap_std(beta_pred, bootstrap)
        beta_pred_bootstrap = np.sum(beta_pred)/bootstrap
    # else:
    #     result = minimize_by_scipy(x, dslope_noprob_law_likelihood, (alpha1_0, alpha2_0, x_c0, beta0))
    #     alpha1_pred, alpha2_pred, x_c_pred, beta_pred = result.x[0], result.x[1], result.x[2], result.x[3]

    result = minimize_by_scipy(x, dslope_noprob_law_likelihood, (alpha1_0, alpha2_0, x_c0, beta0), bounds=((2, 6), (0.5, 2), (0.00001, None), (0.000001, 0.999999)))
    alpha1_pred, alpha2_pred, x_c_pred, beta_pred = result.x[0], result.x[1], result.x[2], result.x[3]
    cp1 = time.perf_counter()
    hess = dslope_power_no_prob_hessian(x, alpha1_pred, alpha2_pred, x_c_pred, beta_pred)
    cp2 = time.perf_counter()
    print(f'Time ellapsed to Compute the Hessian: {round(1000 * (cp2-cp1), 3)}ms')
    alpha1_std, alpha2_std, x_c_std, beta_std = np.diagonal(np.sqrt(abs(np.linalg.inv(hess))))
    akaike = akaike_information(x, dslope_noprob_law_likelihood, (alpha1_pred, alpha2_pred, x_c_pred, beta_pred), x_min=x_min)
    bayes = bayesian_information(x, dslope_noprob_law_likelihood, (alpha1_pred, alpha2_pred, x_c_pred, beta_pred), x_min=x_min)
    
    x_estimated = np.linspace(min(x), max(x), len(x))
    px_bootstrap, tau_b = dslope_power_pdf(x_estimated, alpha1=alpha1_pred_bootstrap, alpha2=alpha2_pred_bootstrap, x_c=x_c_pred_bootstrap, beta=beta_pred_bootstrap, x_min=x_min)
    px, tau = dslope_power_pdf(x_estimated, alpha1=alpha1_pred, alpha2=alpha2_pred, x_c=x_c_pred, beta=beta_pred, x_min=x_min)
    
    errors_non_bootstrap = []
    errors_bootstrap = []
    for i in range(16):
        alpha12test = max(2, alpha1_pred + (-1)**(i//1)*alpha1_std)
        alpha22test = min(alpha12test, max(0.5, alpha2_pred + (-1)**(i//2)*alpha2_std))
        xc2test = max(1, x_c_pred + (-1)**(i//4)*x_c_std)
        beta2test = min(1, max(0, beta_pred + (-1)**(i//8)*beta_std))
        errors_non_bootstrap.append(dslope_power_pdf(x_estimated, alpha1=alpha12test, alpha2=alpha22test, x_c=xc2test, beta=beta2test, x_min=x_min)[0])

        alpha12test_b = max(2, alpha1_pred_bootstrap + (-1)**(i//1)*alpha1_std_bootstrap)
        alpha22test_b = min(alpha12test_b, max(0.5, alpha2_pred_bootstrap + (-1)**(i//2)*alpha2_std_bootstrap))
        xc2test_b = max(1, x_c_pred_bootstrap + (-1)**(i//4)*x_c_std_bootstrap)
        beta2test_b = min(1, max(0, beta_pred_bootstrap + (-1)**(i//8)*beta_std_bootstrap))
        errors_bootstrap.append(dslope_power_pdf(x_estimated, alpha1=alpha12test_b, alpha2=alpha22test_b, x_c=xc2test_b, beta=beta2test_b, x_min=x_min)[0])

    errors_non_bootstrap = np.array([np.max(np.array(errors_non_bootstrap), axis=0), np.min(np.array(errors_non_bootstrap), axis=0)])
    errors_bootstrap = np.array([np.max(np.array(errors_bootstrap), axis=0), np.min(np.array(errors_bootstrap), axis=0)])

    errors = np.array([errors_non_bootstrap, errors_bootstrap])
    # Plot
    if show_data:
        print(f'Akaike Information Criteria: {round(akaike, 4)}\tBayesian Information Criteria: {round(bayes, 4)}')
        print(f'\nNon-Bootstrapped Samples:\n\tSlope 1 Predicted: {alpha1_pred} +- {alpha1_std}\n\tSlope 2 Predicted: {alpha2_pred} +- {alpha2_std}\n\tDamping Factor Predicted: {x_c_pred} +- {x_c_std}\n\tProbability Predicted: {beta_pred} +- {beta_std}\n\tCrossing Point Predicted: {tau}')
        print(f'\nBootstrapped Samples:\n\tSlope 1 Predicted: {alpha1_pred_bootstrap} +- {alpha1_std_bootstrap}\n\tSlope 2 Predicted: {alpha2_pred_bootstrap} +- {alpha2_std_bootstrap}\n\tDamping Factor Predicted: {x_c_pred_bootstrap} +- {x_c_std_bootstrap}\n\tProbability Predicted: {beta_pred_bootstrap} +- {beta_std_bootstrap}\n\tCrossing Point Predicted: {tau_b}')
        plot_data(x, n_bins, x_estimated, px_bootstrap, px, errors, title='Double Power Law with Exponential Cutoff and Unknown Pob Distribution')

    return alpha1_pred, alpha2_pred, x_c_pred, beta_pred

# ----------------------------- Statistic Toolkit -----------------------------
def bootstrap_sample(data:np.array):
    """
    Function to return a bootstrapped sample.
    Input: data: an array of values.
    Output: bootstrapped sample: a random sample of size n where data can be repeated.
    """  
    return np.random.choice(data, size=len(data))

def bootstrap_std(estimator:float, bootstrap:int):
    """
    Function to return the bootstrap standard error.
    Input: Estimator: an array of bootstrap estimations of the parameter.
    Output: bootstrap std: the bootstrap standard error for a given estimator and a given number of boostrapped samples.
    """  
    return np.sqrt(np.sum((estimator - np.sum(estimator)/bootstrap)**2)/(bootstrap - 1))

def akaike_information(x:np.array, likelihood, params:tuple, x_min:float=1):
    l = likelihood(params, x, x_min=x_min)
    return 2 * len(params) + 2 * l + (2*len(params)**2 + 2*len(params))/(len(x) - len(params) - 1)

def bayesian_information(x:np.array, likelihood, params:tuple, x_min:float=1):
    l = likelihood(params, x, x_min=x_min)
    return np.log(len(x)) * len(params) + 2 * l

# ----------------------------- Mathematical Toolkit -----------------------------
def d_uppergamma(x, alpha, x_c, x_min=1):
    vuppergamma = np.vectorize(lambda s,x: np.float64(uppergamma(s, x)))
    alpha_step = 1e-6
    up_gamma_plus, up_gamma_minus, up_gamma = vuppergamma(1-alpha+alpha_step, x/x_c), vuppergamma(1-alpha-alpha_step, x/x_c), vuppergamma(1-alpha, x/x_c)

    exp_x = np.exp(-x/x_c)
    dgamma_alpha = (up_gamma_plus - up_gamma_minus)/ alpha_step
    dgamma_x_c = exp_x * x**(1-alpha) / (x_c**(2-alpha))
    d_gamma = np.array([dgamma_alpha, dgamma_x_c])

    d2gamma_alpha_alpha = (up_gamma_plus + up_gamma_minus - 2*up_gamma)/ alpha_step**2
    d2gamma_alpha_x_c = - np.log(x/x_c) * exp_x * x**(1-alpha) / (x_c**(2-alpha))
    d2gamma_x_c_x_c = (x_c * (alpha - 2) + x) * exp_x * x**(1-alpha) / (x_c**(4-alpha))
    d2gamma = np.array([[d2gamma_alpha_alpha, d2gamma_alpha_x_c],
                        [d2gamma_alpha_x_c, d2gamma_x_c_x_c]])
    return (d_gamma, d2gamma)
    
if __name__ == '__main__':
    n_samples = 10000
    n_bins = 100

    # generate_power_law_sample(n_samples=n_samples, alpha=2, x_min=1, show_data=True, n_bins=n_bins, bootstrap=100) # Bootstrap has no effect as the parameters are analytic
    # generate_cutoff_law_sample(n_samples=n_samples, alpha=2, x_c=100, x_min=1, show_data=True, n_bins=n_bins, bootstrap=100)
    # generate_dslope_law_sample(n_samples=n_samples, alpha1=4, alpha2=1.5, x_c=100, x_min=1, beta=0.5, show_data=True, n_bins=n_bins, bootstrap=100)
    # generate_dslope_noprob_law_sample(n_samples=n_samples, alpha1=4, alpha2=1.5, x_c=100, beta=0.5, x_min=1, show_data=True, n_bins=n_bins, bootstrap=100)

    import pandas as pd
    data2 = pd.read_csv('data/c2/13_23_bzr_events_c2.dat', sep='\s', header=None)
    x2 = data2[1].to_numpy()
    len(x2)
    generate_dslope_law_sample(x=x2, alpha1=2, alpha2=1.2, x_c=100, beta=0.8, x_min=1, show_data=True, n_bins=30, bootstrap=100)
