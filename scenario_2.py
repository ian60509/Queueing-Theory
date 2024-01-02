import numpy as np
import matplotlib.pyplot as plt
from sympy import *

def theoretical_waiting_cdf(arrival_rate, service_rate_1, service_rate_2, waiting_times, num_packets, case_1_ratio):
    
    mu1 = service_rate_1
    mu2 = service_rate_2
    Lambda = arrival_rate
    p1 = case_1_ratio
    p2 = 1-case_1_ratio
    rho = p1*(Lambda / mu1) + p2*(Lambda / mu2)
    
    s = Symbol('s')
    t = Symbol('t')
    B_Laplace = p1 * (mu1 / (s + mu1)) + p2 * (mu2 / (s + mu2))
    W_Laplace = (1 - rho) / (s - Lambda + Lambda * B_Laplace)
    w_y = inverse_laplace_transform(W_Laplace, s, t)

    waiting_time_theo_cdf = np.zeros(num_packets)
    for i, y in enumerate(waiting_times):
        waiting_time_theo_cdf[i]  = w_y.subs(t, y) 

    return waiting_time_theo_cdf
def theoretical_system_cdf(arrival_rate, service_rate_1, service_rate_2, system_times, num_packets, case_1_ratio):
    mu1 = service_rate_1
    mu2 = service_rate_2
    Lambda = arrival_rate
    p1 = case_1_ratio
    p2 = 1-case_1_ratio
    rho = p1*(Lambda / mu1) + p2*(Lambda / mu2)

    s = Symbol('s')
    t = Symbol('t')
    B_Laplace_s = p1 * (mu1 / (s + mu1)) + p2 * (mu2 / (s + mu2))
    W_Laplace_s = s * (1 - rho) / (s - Lambda + Lambda * B_Laplace_s)
    S_Laplace_s = B_Laplace_s * W_Laplace_s
    S_y = inverse_laplace_transform(S_Laplace_s, s, t)
    S_integrate = integrate(S_y)

    system_time_theo_cdf = np.zeros(num_packets)
    
    for i in range(len(system_times)):
        y = system_times[i]
        system_time_theo_cdf[i] = S_integrate.subs(t, y) 

    return system_time_theo_cdf
def simulate_queue(arrival_rate, service_rate_1, service_rate_2, num_packets, case_1_ratio):
    arrival_times = np.cumsum(np.random.exponential(1/arrival_rate, size=num_packets))
    num_packets_1 = (int)(num_packets * case_1_ratio)
    num_packets_2 = (int)(num_packets * (1-case_1_ratio))
    service_times_1 = np.random.exponential(1/service_rate_1, size=num_packets_1)
    service_times_2 = np.random.exponential(1/service_rate_2, size=num_packets_2)
    service_times = np.concatenate((service_times_1, service_times_2))
    np.random.shuffle(service_times) 

    departure_times = np.zeros_like(arrival_times)
    waiting_times = np.zeros_like(arrival_times)
    system_times = np.zeros_like(arrival_times)

    for i in range(num_packets):
        if i==0:
            start_service_time = arrival_times[i]
        else:
            start_service_time = max(departure_times[i-1],arrival_times[i] )
            # print("start service time = ",start_service_time, "i=", i)
        
        waiting_times[i] = start_service_time - arrival_times[i]
        departure_times[i] = service_times[i] + start_service_time
        system_times[i] = service_times[i] + waiting_times[i]

    waiting_times = sorted(waiting_times)
    system_times = sorted(system_times)

    return waiting_times, system_times
def calculate_mse(simulated_cdf, theoretical_cdf):
    return np.mean((simulated_cdf - theoretical_cdf)**2)
def plot_cdf(x_values, cdf_simulated, cdf_theoretical, title):
    
    cdf_theoretical = np.unique(cdf_theoretical)
    cdf_simulated = np.unique(cdf_simulated)
    x_values = np.unique(x_values)
    # print(cdf_simulated.shape,"  ", x_values.shape, "", cdf_theoretical.shape)
    plt.step(x_values, cdf_simulated, label='Simulated')
    plt.step(x_values, cdf_theoretical, label='Theoretical', linestyle='dashed')
    plt.xlabel('Time')
    plt.ylabel('CDF')
    plt.yticks(np.linspace(0,1,11))
    plt.ylim(0,1.1)
    plt.title(title+f'  lambda={arrival_rate}, p1={case_1_ratio}')
    plt.legend()
    plt.show()
def tranform_to_cdf(distribution):
    cdf = np.zeros(num_packets)
    for i in range(len(distribution)):
        cdf[i] = np.sum(distribution <= distribution[i]) / num_packets
    return cdf
    
# Simulation parameters
num_packets = 100
mu = 10  # λ
Lambda = 10  # μ
arrival_rate = Lambda
service_rate_1 = mu 
service_rate_2 = 2 * mu
case_1_ratio = 0.25


# Simulation
waiting_times, system_times = simulate_queue(arrival_rate, service_rate_1, service_rate_2, num_packets, case_1_ratio)
waiting_time_sim_cdf = tranform_to_cdf(waiting_times)
system_time_sim_cdf = tranform_to_cdf(system_times)

# Theoretical CDF
waiting_time_theo_cdf = theoretical_waiting_cdf(arrival_rate, service_rate_1, service_rate_2, waiting_times, num_packets, case_1_ratio)
system_time_theo_cdf = theoretical_system_cdf(arrival_rate, service_rate_1, service_rate_2, system_times, num_packets, case_1_ratio)

# Plotting
# waiting_time_theo_cdf = waiting_time_sim_cdf
plot_cdf(waiting_times, waiting_time_sim_cdf, waiting_time_theo_cdf, 'Waiting Time CDF')
plot_cdf(system_times, system_time_sim_cdf, system_time_theo_cdf, 'System Time CDF')

# Calculate and print MSEs
print(f'waiting time simulaiton={waiting_time_sim_cdf}')
print(f'waiting time theo={waiting_time_theo_cdf}')
waiting_time_mse = calculate_mse(waiting_time_sim_cdf, waiting_time_theo_cdf)
system_time_mse = calculate_mse(system_time_sim_cdf, system_time_theo_cdf)

print(f'Waiting Time MSE: {waiting_time_mse}')
print(f'System Time MSE: {system_time_mse}')
