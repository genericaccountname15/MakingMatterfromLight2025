"""
visualisation.py

Defines the Visualiser class which inherits from the Simulation class.
Uses matplotlib to render the simulation.

Timothy Chew
1/8/25
"""
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from core.simulation import Simulation      #pylint: disable=import-error
from core.simulation import Xray            #pylint: disable=import-error
from core.simulation import Gamma           #pylint: disable=import-error

class Visualiser(Simulation):
    """Visualiser for simulation
    Child of Simulation

    Attributes:
        bath_vis (bool):
            - True to focus on X-ray pulse
            - False to focus on gamma pulse
            Affects the limits of the plot (for values used in the 2018 experiment)
    
    Methods:
        plot: Plots out the simulation with a time step slider
    """
    def __init__(
            self,
            xray_bath: Xray,
            gamma_pulse: Gamma,
            bath_vis=False
        ):
        super().__init__(xray_bath, gamma_pulse)
        self.bath_vis = bath_vis

    ############ METHODS ##########################################################################
    def plot(self):
        """
        Plots out the simulation with a time step slider using matplotlib
        """
        # setting up figure #####################################
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.25) #making space for sliders
        ax.set_title('Gamma and X-ray simulation')
        ax.set_xlabel('x/mm')
        ax.set_ylabel('y/mm')

        if self.get_bath_vis():
            ax.set_xlim(-500, 500)
            ax.set_ylim(0,500)
            t_max = 1000
        else:
            ax.set_xlim(-30, 30)
            ax.set_ylim(0,6)
            t_max = 20

        # plotting coordinates and objects #####################
        xray_coords = self.get_xray_bath().get_xray_coords()

        ax.add_patch( self.get_gamma_pulse().get_gamma_axes_obj() )

        xray_bath, = ax.plot(xray_coords[:, 0], xray_coords[:, 1], 'o', label = 'X-ray bath')

        # check for collisions #################################
        overlap_coords = self.get_overlap_coords(xray_coords, self.get_gamma_pulse().get_bounds())
        if len(overlap_coords) != 0:
            overlap, = ax.plot(overlap_coords[:, 0], overlap_coords[:, 1], 'o', label="overlap")
        else:
            overlap, = ax.plot(0,-10,'o', label='overlap')


        # legend ##############################################
        ax.set_aspect('equal')
        ax.legend(loc = 'upper right')

        # time step ###########################################################################
        # initialising slider #################################
        time_slider = Slider(
            ax = fig.add_axes([0.25, 0.1, 0.65, 0.03]),
            label = 'time(ps)',
            valmin = 0,
            valmax = t_max,
            valinit = 0
        )

        # slider update function #############################
        def update(val): # pylint: disable=unused-argument
            #note: time is in pico seconds and distance in mm
            # move gamma pulse and xrays ##########
            dx = time_slider.val * 1e-12 * 3e8 * 1e3
            self.gamma_pulse.move(dx)
            xray_coords_moved  = self.xray_bath.move_xrays(time_slider.val)

            xray_bath.set_xdata(xray_coords_moved[:, 0])
            xray_bath.set_ydata(xray_coords_moved[:, 1])

            # check for overlapping ##############
            overlap_coords = self.get_overlap_coords(
                xray_coords_moved, self.get_gamma_pulse().get_bounds())

            if len(overlap_coords) != 0:
                overlap.set_xdata(overlap_coords[:, 0])
                overlap.set_ydata(overlap_coords[:, 1])
            else:
                overlap.set_xdata([0])
                overlap.set_ydata([-10])

            fig.canvas.draw_idle()

        time_slider.on_changed(update)


        plt.show()

    ############ ACCESS METHODS ####################################################################
    def get_bath_vis(self) -> bool:
        """Access method for bath visualiser boolean

        Returns:
            bool: bath visualiser boolean
        """
        return self.bath_vis