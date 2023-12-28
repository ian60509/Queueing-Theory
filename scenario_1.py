import numpy as np
import matplotlib.pyplot as plt

def theoretical_waiting_cdf(arrival_rate, service_rate, waiting_times, num_packets):
    rho = arrival_rate / service_rate
    mu = service_rate
    Lambda = arrival_rate

    max_waiting_time = np.max(waiting_times)

    waiting_time_theo_cdf = np.zeros(num_packets)
    
    for i in range(len(waiting_times)):
        y = waiting_times[i]
        waiting_time_theo_cdf[i] = 1 - rho * np.exp(-mu*(1-rho)*y)

    return waiting_time_theo_cdf
def theoretical_system_cdf(arrival_rate, service_rate, system_times, num_packets):
    rho = arrival_rate / service_rate
    mu = service_rate
    Lambda = arrival_rate
    system_time_theo_cdf = np.zeros(num_packets)
    
    for i in range(len(system_times)):
        y = system_times[i]
        system_time_theo_cdf[i] = 1 - np.exp(-mu*(1-rho)*y)

    return system_time_theo_cdf

def simulate_queue(arrival_rate, service_rate, num_packets):
    arrival_times = np.cumsum(np.random.exponential(1/arrival_rate, size=num_packets))
    service_times = np.random.exponential(1/service_rate, size=num_packets)

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
    plt.title(title)
    plt.legend()
    plt.show()
def tranform_to_cdf(distribution):
    cdf = np.zeros(num_packets)
    for i in range(len(distribution)):
        cdf[i] = np.sum(distribution <= distribution[i]) / num_packets
    return cdf
# Simulation parameters
arrival_rate = 8  # λ
service_rate = 10  # μ
num_packets = 10000



# Simulation
waiting_times, system_times = simulate_queue(arrival_rate, service_rate, num_packets)
waiting_time_sim_cdf = tranform_to_cdf(waiting_times)
system_time_sim_cdf = tranform_to_cdf(system_times)

# Theoretical CDF
x_values = np.linspace(0, 10, num_packets) #set 0~10 fixed interval list
waiting_time_theo_cdf = theoretical_waiting_cdf(arrival_rate, service_rate, waiting_times, num_packets)
system_time_theo_cdf = theoretical_system_cdf(arrival_rate, service_rate, system_times, num_packets)

# Plotting
plot_cdf(waiting_times, waiting_time_sim_cdf, waiting_time_theo_cdf, 'Waiting Time CDF')
plot_cdf(system_times, system_time_sim_cdf, system_time_theo_cdf, 'System Time CDF')

# Calculate and print MSEs
waiting_time_mse = calculate_mse(waiting_time_sim_cdf, waiting_time_theo_cdf)
system_time_mse = calculate_mse(system_time_sim_cdf, system_time_theo_cdf)

print(f'Waiting Time MSE: {waiting_time_mse}')
print(f'System Time MSE: {system_time_mse}')
