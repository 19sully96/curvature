# curvature.py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numba import jit



# load WMAP TT Power Spectrum data
    # https://lambda.gsfc.nasa.gov/data/map/powspec/map_comb_tt_powspec_yr1_v1p1.txt
print('Loading data...')
ell = np.loadtxt('map_comb_tt_powspec_yr1_v1p1.txt',usecols=(0))
spectrum = np.loadtxt('map_comb_tt_powspec_yr1_v1p1.txt',usecols=(1))
spectrum_error = np.loadtxt('map_comb_tt_powspec_yr1_v1p1.txt',usecols=(2))
print('Data successfully loaded.\n')



# Create Power Spectrum plot
print('Creating initial power spectrum plot...')
fig, ax = plt.subplots(1,1,figsize=(10,8))
ax.plot(ell[0:500],spectrum[0:500],'.')
ax.set_xlabel('$\ell$', fontsize=22)
ax.set_ylabel('Temperature [$\mu K^2$]', fontsize=22)
ax.set_title('WMAP Combined TT Power Spectrum (1 year)', fontsize=22)
ax.tick_params(width=2,length=7,labelsize=15)
fig.savefig('power_spectrum.png')
print('Plot created.\n')



# Defining functions for MCMC simulated annealing fitting

# gaussian function
@jit(nopython=True)
def gauss(x,param):
    A = param[0]
    mu = param[1]
    sigma = param[2]
    return A*np.exp(-(x - mu)**2 / (2 * sigma**2))

# error function
@jit(nopython=True)
def error(data,expected):
    numerator = ((data - expected)**2)
    denominator = expected
    array_to_sum = numerator / denominator
    err = np.sum(array_to_sum)
    return err

# MCMC sampler pdf 
@jit(nopython=True)
def prob(ERROR,T):
    return np.exp(-1/T * ERROR)

# proposal distrubution
@jit(nopython=True)
def move(param):
    A, mu, sigma = param[0], param[1], param[2]
    A_std_dev = 5
    mu_std_dev = 1
    sigma_std_dev = 1
    A_new = np.random.normal(loc = A, scale = A_std_dev)
    mu_new = np.random.normal(loc = mu, scale = mu_std_dev)
    sigma_new = np.random.normal(loc = sigma, scale = sigma_std_dev)
    param_new = [A_new, mu_new, sigma_new]
    return param_new
        
# cooling schedule
# @jit(nopython=True)
def temp(step):
    T0 = 1e3
    tau = 1e4
    qv = 2
    if step < (1.9 * 1e5):
        T = T0
    else:
#         T = T0 * (2**(qv - 1) - 1) / ((1 + step)**(qv - 1) - 1)
        T = T0 * np.exp(-(step/tau))
    return T
    
# MCMC 
def fit(param_start, x, data, l_start, l_end):
                                          # Initial values for MCMC
    steps = int(2e5)
#     x = ell[l_start:l_end]
#     data = spectrum[l_start:l_end]


                                          # Metropolis-Hastings Algorithm
    param = param_start
    expected = gauss(x, param)
    ERROR = error(data, expected)
    T = temp(0)
    prob_old = prob(ERROR, T)
    chain = []
    chain.append(param_start)
    error_chain = []
    error_chain.append(ERROR)

    for s in np.arange(steps-1)+1:
        T = temp(s)                        # update temperature

        param_new = move(param)            # propose move

        expected = gauss(x,param_new)      # calculate new probability
        ERROR = error(data, expected)
        prob_new = prob(ERROR,T)            

        r = np.random.rand(1)              # determine if move is accepted
        if (prob_new/prob_old > r):
            prob_old = prob_new
            param = param_new

        A, mu, sigma = param[0], param[1], param[2]
        chain.append([A, mu, sigma])       # update chain
        error_chain.append(ERROR)
        
    return chain[-1], chain, error_chain



# Main Program 
    #iterates MCMC nit times and averages model parameters

print('Starting main fitting procedure...\n')
    
nit = 10                        # number of iterations of MCMC
param_start = [5580, 220, 100]  # initial starting parameters
l_start = 50                    # start of data fitting range
l_end = 400                     # end of data fitting range
x = ell[l_start:l_end]          # x data
data = spectrum[l_start:l_end]  # y data

print('Number of iterations to complete: ', nit)

amplitude = np.zeros(nit)
mean = np.zeros(nit)
variance = np.zeros(nit)

for n in np.arange(nit):
    print('Current iteration: ', n+1)
    param_fit, chain, error_chain = fit(param_start, x, data, l_start, l_end)
    amplitude[n] = param_fit[0]
    mean[n] = param_fit[1]
    variance[n] = param_fit[2]

best_param = [np.mean(amplitude), np.mean(mean), np.mean(variance)]    

print('----------------------------------------\n----------------------------------------')
print('Best Parameter Estimates:')
print('Amplitude = ',np.mean(amplitude), '+/-', np.std(amplitude))
print('Mean = ',np.mean(mean), '+/-', np.std(mean))
print('Variance = ',np.mean(variance), '+/-', np.std(variance))
print('----------------------------------------\n----------------------------------------')

# putting 3-parameter chains (from last iteration) into 1-parameter arrays for plotting
size = 200000
amplitude_chain = np.zeros(size)
mean_chain = np.zeros(size)
variance_chain = np.zeros(size)
for i in np.arange(size):
    amplitude_chain[i] = chain[i][0]
    mean_chain[i] = chain[i][1]
    variance_chain[i] = chain[i][2]




# Creating main plots

print('Creating main plots...')

# Create fitted power spectrum plot
fitted = gauss(ell[l_start:l_end],best_param)
fig, ax = plt.subplots(1,1, figsize=(10,8))
ax.plot(ell[0:500], spectrum[0:500],'.',label='data')
ax.plot(ell[l_start:l_end], fitted, linewidth=3, label='MCMC fit')
ax.set_xlabel('$\ell$', fontsize=22)
ax.set_ylabel('Temperature [$\mu K^2$]', fontsize=22)
ax.set_title('WMAP Combined TT Power Spectrum (1 year)', fontsize=22)
ax.tick_params(width=2,length=7,labelsize=15)
ax.legend(fontsize=20)
fig.savefig('fitted_power_spectrum.png')

# Creating amplitude chain plot
fig, ax = plt.subplots(1,1, figsize=(10,8))
ax.plot(np.arange(np.size(mean_chain)), amplitude_chain, label='MCMC simulation')
ax.axvline(x=190000, color='k', linestyle='--',alpha=0.5, label='Burnout')
ax.set_xlabel('Monte Carlo step', fontsize=22)
ax.set_ylabel('A [$\mu K^2$]',fontsize=22)
ax.tick_params(width=2, length=7, labelsize=15)
ax.legend(fontsize=15)
fig.savefig('amplitude_chain.png')

# Creating mean chain plot
fig, ax = plt.subplots(1,1, figsize=(10,8))
ax.plot(np.arange(np.size(mean_chain)), mean_chain, label='MCMC simulation')
ax.axvline(x=190000, color='k', linestyle='--',alpha=0.5, label='Burnout')
ax.set_xlabel('Monte Carlo step', fontsize=22)
ax.set_ylabel('$\mu$ [$\ell$]',fontsize=22)
ax.tick_params(width=2, length=7, labelsize=15)
ax.legend(fontsize=15)
fig.savefig('mean_chain.png')

# Creating variance chain plot
fig, ax = plt.subplots(1,1, figsize=(10,8))
ax.plot(np.arange(np.size(mean_chain)), variance_chain, label='MCMC simulation')
ax.axvline(x=190000, color='k', linestyle='--',alpha=0.5, label='Burnout')
ax.set_xlabel('Monte Carlo step', fontsize=22)
ax.set_ylabel('$\sigma^2$ [$\ell^2$]',fontsize=22)
ax.tick_params(width=2, length=7, labelsize=15)
ax.legend(fontsize=15)
fig.savefig('variance_chain.png')

# Creating error function chain plot
fig, ax = plt.subplots(1,1, figsize=(10,8))
ax.plot(np.arange(np.size(error_chain)), error_chain, label='MCMC simulation')
ax.axvline(x=190000, color='k', linestyle='--',alpha=0.5, label='Burnout')
ax.set_xlabel('Monte Carlo step', fontsize=22)
ax.set_ylabel('Error function [$\mu K^2$]',fontsize=22)
ax.tick_params(width=2, length=7, labelsize=15)
ax.legend(fontsize=15)
fig.savefig('error_chain.png')

print('Main plots created.\n')




# Unit testing
    # fitting noisy guassian data
def unit_test(x, A, mu, var, noise, epsilon):
    
    print('Starting unit test... \n')
    print('Input variables:')
    print('Amplitude = ',A)
    print('Mean = ', mu)
    print('Variance = ', var)
    print('Noise = ', noise)
    print('Epsilon = ', epsilon, '\n')
    
    param_start = [6000,200,120]
    l_start = 0
    l_end = -1
    
    data = gauss(x, [A,mu,var])    # creating data from input parameters
    i = 0
    for d in data:
        dx = np.random.normal(loc=0, scale=noise)
        data[i] = d + dx           # adding noise
        i = i + 1

                                   # getting model parameters for best fit
    param_fit, chain, error_chain = fit(param_start, x, data, l_start, l_end)
    
                                   # comparing parameters to epsilon
    if np.abs(param_fit[0] - A) > epsilon[0]:
        print('Amplitude FAILED: Exceeds corresponding epsilon')
    else:
        print('Amplitude PASSED')
    if np.abs(param_fit[1] - mu) > epsilon[1]:
        print('Mean FAILED: Exceeds corresponding epsilon')
    else:
        print('Mean PASSED')
    if np.abs(param_fit[2] - var) > epsilon[2]:
        print('Variance FAILED: Exceeds corresponding epsilon \n')
    else:
        print('Variance PASSED \n')

                               # creating fitted plots
    fig,ax=plt.subplots(1,1,figsize=(10,8))
    ax.plot(x, data, '.', label='test data')
    ax.plot(x, gauss(x,param_fit), label='Fitted model')
    ax.set_xlabel('x', fontsize=22)
    ax.set_ylabel('y', fontsize=22)
    ax.set_title('Unit test',fontsize=22)
    ax.tick_params(width=2, length=7, labelsize=15)
    ax.legend(fontsize=15)
    fig.savefig('unit_test.png')
    
    print('Fitted model parameters:')
    print('Amplitude = ',param_fit[0])
    print('Mean = ',param_fit[1])
    print('Variance = ',param_fit[2])
        
        

# Executing unit test
x = np.linspace(10,500,500)
A = 5500
mu = 220
var = 100
noise = 100
epsilon = [300,5,10]
unit_test(x,A,mu,var,noise,epsilon)
