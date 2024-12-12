import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

class Square_membrane_solver:

    def __init__(self, Nx, Ny, a):
        self.x = np.linspace(0, a, Nx)
        self.y = np.linspace(0, a, Ny)
        self.a = a
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.f_values = self.f(self.X, self.Y)  # Values at each grid point, faster than prev when I calculated current value at each iteration

    def f(self, x, y):  # using something similar to exercise 16, where function 0 at limits. more modes means multiply in sin with 2,3,4 etc
        return np.sin(1 * np.pi * x / self.a) * np.sin(1 * np.pi * y / self.a)

    def G_nm(self, n, m):
        # Compute the Fourier coefficients
        integrand = self.f_values * np.sin(np.pi * self.X * n / self.a) * \
                    np.sin(np.pi * self.Y * m / self.a)
        integral = np.sum(integrand) * self.dx * self.dy
        G = (4 / self.a**2) * integral  # Include normalization#########
        #print(f"G({n},{m}) = {G}")  # Debugging Fourier coefficients
        return G

    def psi(self, c, t, N_terms=10):  # Depends on (x, y, t)
        res = np.zeros_like(self.X)

        for n_idx in range(1, N_terms + 1):
            for m_idx in range(1, N_terms + 1):
                omega = (np.pi * c / self.a) * np.sqrt(n_idx**2 + m_idx**2)
                G = self.G_nm(n_idx, m_idx)
                res += G * np.sin(np.pi * n_idx * self.X / self.a) * \
                       np.sin(np.pi * m_idx * self.Y / self.a) * \
                       np.cos(omega * t)

        return res


if __name__ == "__main__":
    # Parameters
    a = 10  # Side length (square)
    Nx, Ny = 100, 100  # Grid resolution
    c = 5.0  # Wave velocity
    t = 0.5  # Time point
    N_terms = 10  # Number of Fourier terms (the sum in my paper solution)

    # 3D parameters
    num_frames = 100  # Number of frames for the animation
    t_duration = 2.0  # Duration of the animation (seconds)

    # init the solver
    solver = Square_membrane_solver(Nx, Ny, a)

    """
    # Test psi at different times
    for t_test in [0, 0.5, 1.0]:
        psi_test = solver.psi(c, t_test, N_terms)
        print(f"t = {t_test}: max displacement = {np.max(psi_test)}, min displacement = {np.min(psi_test)}")
        plt.figure()
        plt.contourf(solver.X, solver.Y, psi_test, cmap='viridis')
        plt.colorbar(label='Displacement')
        plt.title(f"Membrane displacement at t = {t_test}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
    """
    # Setup the figure and 3D axis
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, a)
    ax.set_ylim(0, a)
    ax.set_zlim(-0.5, 0.5)  # Adjust based on psi values
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Displacement")
    ax.set_title("3D Animation of a Vibrating Square Membrane")

    # Initial surface plot
    t_values = np.linspace(0, t_duration, num_frames)
    psi_initial = solver.psi(c, t_values[0], N_terms)
    surface = ax.plot_surface(solver.X, solver.Y, psi_initial, cmap='viridis')

    # Update function for the animation
    def update(frame):
        global surface

        surface.remove()
        psi_t = solver.psi(c, t_values[frame], N_terms)
        #print(f"Frame {frame}: max displacement = {np.max(psi_t)}, min displacement = {np.min(psi_t)}")  # Debugging animation
        surface = ax.plot_surface(solver.X, solver.Y, psi_t, cmap='viridis')


    # Create the animation
    anim = FuncAnimation(fig, update, frames=num_frames, interval=100)
    plt.show()



    anim.save("membrane_animation.mp4", writer="ffmpeg", fps=20)
