import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def energy_and_psi(N, V, h_bar, m, dx):

    # Diagonal
    d = [((h_bar ** 2) / (m * (dx ** 2))) + v for v in V]

    # Over and under diagonal
    e = -(h_bar ** 2) / (2 * m * (dx ** 2))

    H = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                H[i][j] = d[i]
            if abs(i - j) == 1:
                H[i][j] = e

    # Create energy and psi matrix
    energy, psi = np.linalg.eigh(H)
    return energy, psi


def initial_state(k0, x, x0, sigma):
    # Initialize Gaussian distribution
    gaussian_distribution = np.exp(-((x - x0) ** 2) / (4 * (sigma**2)))
    # Normalization of the Gaussian distribution
    normalization_factor = (2 * np.pi * (sigma**2)) ** (-1/4)
    # To make function quadratic integrate-able
    plane_wave_factor = np.exp(1j * k0 * x)

    return normalization_factor * gaussian_distribution * plane_wave_factor


def c_vector(N, psi, Psi0):
    # Make psi matrix complex
    psi_complex = psi * (1.0 + 0.0j)
    # Create c vector
    c = np.zeros(N, dtype=np.complex128)
    for n in range(N):
        c[n] = np.vdot(psi_complex[:, n], Psi0)

    return c


def delta_x_and_p(N, Nt, dt, x, dx, c, psi, energy, h_bar, sigma, m, p0):
    time = np.zeros(Nt)
    delta_x_analytic1 = np.zeros(Nt)
    delta_x_analytic2 = np.zeros(Nt)
    delta_x_numeric1 = np.zeros(Nt)
    delta_x_numeric2 = np.zeros(Nt)
    delta_p_numeric1 = np.zeros(Nt)
    delta_p_numeric2 = np.zeros(Nt)
    rho_matrix = np.zeros((Nt, N))

    psi_complex = psi * (1.0 + 0.0j)

    for i in range(Nt):

        print("Iteration number: ", i + 1)

        time[i] = i * dt
        Psi_t = np.zeros(N, dtype=np.complex128)

        for n in range(N):
            Psi_t += c[n] * psi_complex[:, n] * np.exp(-1j * energy[n] * time[i] / h_bar)

        rho_t = np.abs(Psi_t) ** 2

        # Add to rho matrix
        rho_matrix[i] = rho_t

        # Calculate numeric delta x
        delta_x_numeric1[i] = dx * np.dot(x ** 2, rho_t)
        delta_x_numeric2[i] = (dx ** 2) * (np.dot(x, rho_t) ** 2)

        # Calculate analytic delta x
        delta_x_analytic1[i] = (sigma**2)
        delta_x_analytic2[i] = ((h_bar**2) * (time[i]**2) / (4 * (m**2) * (sigma ** 2)))

        # Calculate numeric delta p
        delta_p_numeric1[i] = np.dot(((p0 + np.sqrt(2.0 * m * energy)) ** 2), ((np.absolute(c) ** 2) * dx))
        delta_p_numeric2[i] = p0 ** 2

    delta_x_analytic = np.sqrt(delta_x_analytic1 + delta_x_analytic2)
    delta_x_numeric = np.sqrt(delta_x_numeric1 - delta_x_numeric2)
    delta_p_numeric = np.sqrt(delta_p_numeric1 - delta_p_numeric2)

    return time, delta_x_numeric, delta_x_analytic, delta_p_numeric, rho_matrix


def main():
    # Style plot
    plt.rcParams['axes.facecolor'] = '#e6e6e6'
    plt.rcParams['grid.color'] = '#ffffff'
    plt.rcParams['grid.linestyle'] = '-'

    # Iterative constants
    N = 1000                            # Number of steps
    dx = 1.0E-10                        # Step size
    x = np.arange(0.0, N * dx, dx)      # Step vector
    Nt = 300                            # Number of time steps

    # Potential
    # True -> V = 0
    # False -> V = x^2
    if False:
        V = np.zeros(N)
    else:
        V = x**2

    # Physical constants
    h_bar = 1.05E-34
    m = 9.11E-31
    E0 = 1.602E-19
    k0 = np.sqrt(2.0 * m * E0) / h_bar

    # Choose these
    p0 = k0 * h_bar
    sigma = 20.0 * dx
    x0 = 10 * sigma
    dt = 2 * N * dx / ((p0 / m) * 500)

    # energy vector and psi matrix
    energy, psi = energy_and_psi(N, V, h_bar, m, dx)

    # Plot energy
    plt.figure('energy')
    plt.plot([n for n in range(N)], energy, color='#000000')
    plt.title("Energy values")
    plt.xlabel("n (iterations)")
    plt.ylabel("Energy (C)")
    plt.show()

    # Initial state
    Psi0 = initial_state(k0, x, x0, sigma)

    # Plot initial state
    plt.figure('initial-state')
    plt.plot(x, np.abs(Psi0 ** 2), color='#000000')
    plt.title("Probability distribution of the initial state")
    plt.xlabel("$x$ (m)")
    plt.ylabel("$|\Psi(x,0)|^2$")
    plt.show()

    # C vector
    c = c_vector(N, psi, Psi0)

    # Plot c vector
    plt.figure('c-vector')
    plt.plot([n for n in range(N)], np.abs(c) ** 2, color='#000000')
    plt.title("C vector")
    plt.xlabel("n (iterations)")
    plt.ylabel("$|c(n)|^2$")
    plt.show()

    # Calculate delta x (numeric and analytic) and delta p (numeric)
    time, \
    delta_x_numeric, \
    delta_x_analytic, \
    delta_p_numeric, \
    rho_matrix = delta_x_and_p(N, Nt, dt, x, dx, c, psi, energy, h_bar, sigma, m, p0)

    # Plot delta x
    plt.figure('delta-x')
    #plt.plot(time, delta_x_analytic, label="$\Delta x(t)$ analytic")
    plt.plot(time, delta_x_numeric, label="$\Delta x(t)$ numeric")
    plt.title("Standard deviation in position")
    plt.xlabel("$t$ (s)")
    plt.ylabel("$\Delta x(t)$ (m)")
    plt.legend()
    plt.show()

    # Plot delta p
    plt.figure('delta-p')
    plt.plot(time, delta_p_numeric, label="$\Delta p(t)$ numeric")
    plt.title("Standard deviation in momentum")
    plt.xlabel("$t$ (s)")
    plt.ylabel("$\Delta p(t)$ (kg * m / s)")
    plt.show()

    # Plot Heisenberg uncertainty
    plt.figure('heisenberg')
    plt.plot(time, [(h_bar / 2) for i in range(300)], label="Minimum uncertainty")
    plt.plot(time, delta_x_numeric * delta_p_numeric, label="Numerical uncertainty")
    plt.title("Heisenberg uncertainty principle")
    plt.xlabel("$t$ [s]")
    plt.ylabel("$\Delta p \cdot \Delta x$")
    plt.legend()
    plt.show()

    # Animated plot
    fig = plt.figure("Wave packet animation")
    ax = plt.axes(xlim=(0.0 * dx, N * dx), ylim=(0, 1.5 * np.max(np.abs(Psi0) ** 2)))
    line, = ax.plot([], [], lw=1)

    # Calculate delta x (numeric and analytic) and delta p (numeric) again
    time, \
    delta_x_numeric, \
    delta_x_analytic, \
    delta_p_numeric, \
    rho_matrix = delta_x_and_p(N, Nt, 3.0E-15, x, dx, c, psi, energy, h_bar, sigma, m, p0)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        line.set_data(x, rho_matrix[i])
        return line,

    plt.plot(x, V * np.max(np.abs(Psi0) ** 2) / 2)
    plt.xlabel('$x$ (m)')
    anim = animation.FuncAnimation(fig, animate, init_func=init, repeat=True, frames=Nt, interval=20, blit=True)
    plt.show()


main()