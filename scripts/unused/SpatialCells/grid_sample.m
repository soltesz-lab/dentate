
r_off = this_lambda * np.sqrt(local_random.random())
phi_off = local_random.uniform(-np.pi, np.pi)
x_off = r_off * np.cos(phi_off)
y_off = r_off * np.sin(phi_off)

