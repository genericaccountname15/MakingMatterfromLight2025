"""
Simulation of an X-ray gaussian temporal profile and a rectangular Gamma pulse

Timothy Chew
16/7/25
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle

# Xray bath ##################################################################################################
class Xray:
    """
    Coordinates and distribution of the X-ray bath

    Attributes:
        FWHM (float): Full-Width-Half-Maximum of the X-ray distribution (mm)
        Variance (float): Variance of the X-ray distribution (mm^2)
    """
    def __init__(self, FWHM, rotation=0, n_samples_angular=400, n_samples=10):
        self.FWHM = FWHM
        self.variance = ( FWHM / 2.3548 ) ** 2 #convert from FWHM to variance
        self.rotation = rotation
        self.n_samples_angular = n_samples_angular
        self.n_samples = n_samples

        self.xray_coords = self.gen_Xray_seed(
            mean = -self.get_FWHM(),
            variance = self.get_variance(),
            rotation=rotation,
            n_samples_angular = n_samples_angular,
            n_samples = n_samples
        )
    
    ############ METHODS #####################################################################################
    def gen_Xray_seed(self, mean, variance, rotation=0, n_samples_angular = 400, n_samples = 10):
        """Generates distribution of X ray pulse in 2D

        Args:
            mean (float): mean of distribution, radial position (m)
            variance (float): variance of x-ray distribution (mm^2)
            n_samples_angular (int, optional): Number of angles to sample. Defaults to 400
            n_samples (int, optional): Number of samples per angle. Defaults to 10.

        Returns:
            list: list of coordinates for distribution points
        """
        coords = []

        #rotate distribution 180 degrees
        angles = np.linspace(0 + rotation, np.pi + rotation, n_samples_angular)
        for theta in angles:
            ndist = np.random.normal(mean, variance, n_samples) # random distribution centred at 0

            #rotation matrix
            rot_matrix = np.array(
                [ [np.cos(theta), -np.sin(theta)], 
                [np.sin(theta), np.cos(theta)] ])

            #append coords
            for k in ndist:
                rotated_coords = np.matmul(rot_matrix, [k, 0])
                coords.append([rotated_coords[0], rotated_coords[1], theta])
        
        return np.array(coords)
    
    def move_Xrays(self, t):
        """Moves xray points

        Args:
            t (float): Time step (ps)

        Returns:
            numpy.ndarray: moved xray point coordinates
        """
        t *= 1e-9 #mm and picosecond, unit conversion
        moved_coords = []
        for r in self.get_xray_coords():
            #calculate distance to move
            dx = 3e8 * t * np.cos(r[2])
            dy = 3e8 * t * np.sin(r[2])

            moved_coords.append([r[0] + dx, r[1] + dy, r[2]])
        
        return np.array(moved_coords)
    
    def resample(self):
        """Resamples x-ray distribution
        """
        self.xray_coords = self.gen_Xray_seed(
            mean = -self.get_FWHM(),
            variance = self.get_variance(),
            rotation = self.get_rotation(),
            n_samples_angular = self.get_n_samples_angular(),
            n_samples = self.get_n_samples()
        )

    ############ ACCESS METHODS ##############################################################################
    def get_FWHM(self):
        """Access method for FWHM

        Returns:
            float: Full-Width-Half-Maximum of the X-ray distribution (mm)
        """
        return self.FWHM
    
    def get_variance(self):
        """Access method for variance

        Returns:
            float: Variance of the X-ray distribution (mm^2)
        """
        return self.variance

    def get_xray_coords(self):
        """Access method for Xray coordinates

        Returns:
            numpy.ndarray: Array of xray coordinates
        """
        return self.xray_coords
    
    def get_n_samples_angular(self):
        return self.n_samples_angular
    
    def get_n_samples(self):
        return self.n_samples
    
    def get_rotation(self):
        return self.rotation

# Gamma pulse ###################################################################################################
class Gamma:
    """
    Gamma pulse parameters and axes object

    Attributes:
        x_pos (float): x-coordinate of gamma pulse (left edge of rectangle) (mm)
        pulse_length (float): length of gamma pulse (mm)
        height (float): height of gamma pulse in 2D, for an azimuthal angle of 0, this corresponds to the radius (mm)
        off_axis_dist (float): perpendicular distance from bottom of gamma pulse to x-ray source when directly over head (mm)
        facecolor (tuple, optional): face colour of axes object. Defaults to (0, 0, 1, 0.5).
        edgecolor (tuple, optional): edge colour of axes object. Defaults to (1, 0, 0).
        linewidth (int, optional): line width of axes object. Defaults to 2.

    Methods:
        get_bounds: returns the coordinates of the 4 corners of the gamma pulse
    """
    def __init__(self, x_pos, pulse_length, height, off_axis_dist, 
                 facecolor=(0, 0, 1, 0.5), edgecolor=(1, 0, 0), linewidth=2):
        self.x_pos = x_pos
        self.pulse_length = pulse_length
        self.height = height
        self.off_axis_dist = off_axis_dist

        self.gamma_axes_obj = Rectangle(
            xy = [ x_pos, off_axis_dist ] ,
              width = pulse_length ,
              height = height ,
              facecolor = facecolor ,
              edgecolor = edgecolor ,
              linewidth = linewidth ,
              label = 'gamma pulse')

    ############ METHODS #####################################################################################
    def get_bounds(self):
        """Coordinates of the gamma pulse boundaries

        Returns:
            list: coordinates of the 4 corners of the gamma pulse [xmin, xmax, ymin, ymax]
        """
        xmin = self.gamma_axes_obj.get_x()
        ymin = self.gamma_axes_obj.get_y()

        bounds = [xmin, xmin + self.pulse_length,
                  ymin, ymin + self.height]
        
        return bounds
    
    def move(self, dx):
        """Moves gamma pulse object

        Args:
            dx (float): distance to move (mm)
        """
        self.gamma_axes_obj.set_x(self.get_x_pos() + dx)

    def set_x_pos(self, new_x_pos):
        self.x_pos = new_x_pos
        self.gamma_axes_obj.set_x(new_x_pos)

    def set_height(self, new_height):
        self.height = new_height
    
    ############ ACCESS METHODS ##############################################################################
    def get_x_pos(self):
        """Access method for initial x position

        Returns:
            float: initial x-coordinate (mm)
        """
        return self.x_pos
    
    def get_pulse_length(self):
        """Access method for pulse length

        Returns:
            float: length of pulse (mm)
        """
        return self.pulse_length
    
    def get_height(self):
        """Access method for pulse height

        Returns:
            float: pulse height (mm)
        """
        return self.height
    
    def get_off_axis_dist(self):
        """Access method for the axial displacement

        Returns:
            float: axial displacement (mm)
        """
        return self.off_axis_dist
    
    def get_gamma_axes_obj(self):
        """Access method for the gamma axes object

        Returns:
            matplotlib.patches.Rectangle: Axes object for gamma pulse
        """
        return self.gamma_axes_obj
    

# Simulation #################################################################################################
class Simulation:
    """
    Simulation of X-ray bath and gamma pulse interactions (hits)

    Attributes:
        xray_bath (Xray): Xray object instance
        gamma_pulse (Gamma): Gamma object instance
        n_samples_angular (int, optional): Number of angles to sample from 0-pi. Defaults to 400
        n_samples (int, optional): Number of samples per angle. Defaults to 10.
        xray_bath_coords (numpy.ndarray): coordinates of generated Xray bath points
    """
    def __init__(self, xray_bath, gamma_pulse, n_samples_angular = 400, n_samples = 10): #classes
        self.xray_bath = xray_bath
        self.gamma_pulse = gamma_pulse

        self.n_samples_angular = n_samples_angular
        self.n_samples = n_samples
    
    ############ METHODS #####################################################################################
    def get_overlap_coords(self, coords, beam_bounds):
        """Get coordinates of overlapping points of x-ray and gamma pulses

        Args:
            coords (list): Coordinates of x-ray pulse points
            beam_bounds (list): Boundaries of gamma pulse [xmin, xmax, ymin, ymax]

        Returns:
            numpy.ndarray: Array of overlapping coordinates
        """
        overlap_coords = []
        for r in coords:
            if beam_bounds[0] <= r[0] <= beam_bounds[1]:
                if beam_bounds[2] <= r[1] <= beam_bounds[3]:
                    overlap_coords.append([r[0], r[1]])
        
        return np.array(overlap_coords)

    def find_hits(self, eff_height = None, eff_d = None):
        """Calculate hits using X-ray seed coordinates
        Check future position and see if registers a hit

        Args:
            eff_height (float): effective height if on an off-azimuthal position
            eff_d (float): effective off-axis displacement
        
        Returns:
            tuple (float, list): (number of ovelaps, coordinates of overlaps [time, x, y, angle])
        """
        seed_coords = self.get_xray_bath().get_xray_coords()
        bounds = self.get_gamma_pulse().get_bounds()
     
        if eff_d is not None:
            bounds[2] += eff_d

        if eff_height is not None:
            bounds[3] = bounds[2] + eff_height

        #unit conversion back to metres (m)
        beam_bounds = np.array(bounds) * 1e-3
        coords = seed_coords * np.array([1e-3, 1e-3, 1])


        n = 0 #hit counter
        overlap_coords = [] #coordinates of hits [time, x, y, theta]

        for r in coords:
            if r[1] > beam_bounds[3]: #check if already above beam
                pass #skip calculations

            elif r[2] == 0: #check if axial, ignore if so (will have to change if moving beam around)
                pass

            #check if already in beam
            elif beam_bounds[0] <= r[0] <= beam_bounds[1]:
                if beam_bounds[2] <= r[1] <= beam_bounds[3]:
                    n += 1
                    overlap_coords.append([0, r[0], r[1], r[2]])

            else:
                #calculate time to hit max and min boundaries
                t_ymin = ( beam_bounds[2] - r[1]) / ( 3e8 * np.sin( r[2] ) )
                t_ymax = ( beam_bounds[3] - r[1]) / ( 3e8 * np.sin( r[2] ) )

                # future pulse positions
                gamma_xmin_tmin, gamma_xmax_tmin = t_ymin * 3e8 + (beam_bounds[0], beam_bounds[1])
                gamma_xmax_tmax = t_ymax * 3e8 + beam_bounds[1]

                # future x-ray point position
                rx_tmin = t_ymin * 3e8 * np.cos( r[2] ) + r[0]
                rx_tmax = t_ymax * 3e8 * np.cos( r[2] ) + r[0]

                # check inside boundary at t=tmin
                if gamma_xmin_tmin <= rx_tmin <= gamma_xmax_tmin:
                    n += 1
                    overlap_coords.append([t_ymin, rx_tmin, beam_bounds[3], r[2]])
                
                # if past beam, check if enters beam
                elif rx_tmin > gamma_xmax_tmin:
                    if rx_tmax < gamma_xmax_tmax:
                        n += 1
                        x = gamma_xmax_tmax - (gamma_xmax_tmax - rx_tmax)
                        t = t_ymax - (gamma_xmax_tmax - rx_tmax) / 3e8
                        y = beam_bounds[2] + 3e8 * np.sin(r[2]) * (t_ymax - t_ymin)
                        overlap_coords.append([t, x, y, r[2]])

        if len(overlap_coords) == 0:
            return 0, np.array([0,0,0,0])
        else:
            #unit conversion
            overlap_coords = np.array(overlap_coords)
            overlap_coords *= np.array([1e12, 1e3, 1e3, 1])

        return n, overlap_coords

    ############ ACCESS METHODS ##############################################################################
    def get_gamma_pulse(self):
        """Access method for the gamma pulse

        Returns:
            Gamma: Gamma pulse instance
        """
        return self.gamma_pulse

    def get_xray_bath(self):
        """Access method for the x ray path

        Returns:
            Xray: Xray bath instance
        """
        return self.xray_bath

    def get_n_samples_angular(self):
        """Access method for the number of angular samples

        Returns:
            int: number of angles to sample
        """
        return self.n_samples_angular

    def get_n_samples(self):
        """Access method for number of samples

        Returns:
            int: number of points to sample per angle
        """
        return self.n_samples


# visualiser ####################################################################################################
class Visualiser(Simulation):
    """Visualiser for simulation
    Child of Simulation

    Attributes:
        bath_vis (bool): True to focus on X-ray pulse, False to focus on gamma pulse
                        Affects the lims of the plot, works for experimental values only
    
    Methods:
        plot: Plots out the simulation with a time step slider
    """
    def __init__(self, xray_bath, gamma_pulse, n_samples_angular=400, n_samples=10, bath_vis=False):
        super().__init__(xray_bath, gamma_pulse, n_samples_angular, n_samples)
        self.bath_vis = bath_vis

    ############ METHODS #####################################################################################
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
            ax.set_xlim(-4, 4)
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
            xray_coords_moved  = self.xray_bath.move_Xrays(time_slider.val)
            
            xray_bath.set_xdata(xray_coords_moved[:, 0])
            xray_bath.set_ydata(xray_coords_moved[:, 1])

            # check for overlapping ##############
            overlap_coords = self.get_overlap_coords(xray_coords_moved, self.get_gamma_pulse().get_bounds())

            if len(overlap_coords) != 0:
                overlap.set_xdata(overlap_coords[:, 0])
                overlap.set_ydata(overlap_coords[:, 1])
                print('overlap')
            else:
                overlap.set_xdata([0])
                overlap.set_ydata([-10])

            fig.canvas.draw_idle()

        time_slider.on_changed(update)


        plt.show()

    ############ ACCESS METHODS ##############################################################################
    def get_bath_vis(self):
        """Access method for bath visualiser boolean

        Returns:
            bool: bath visualiser boolean
        """
        return self.bath_vis


class Test:
    """
    For running tests on the simulation
    """
    def __init__(self):
        pass

    ############ METHODS #####################################################################################
    def test_values(self):
        """
        Runs the simulation using experiment accurate values
        """
        import values as values
        xray = Xray(
            FWHM = values.xray_FWHM
        )

        gamma = Gamma(
            x_pos = -10e-12 * 3e8 * 1e3,
            pulse_length = values.gamma_length,
            height = values.gamma_radius,
            off_axis_dist = values.off_axial_dist
        )

        vis = Visualiser(
            xray_bath = xray,
            gamma_pulse = gamma,
            bath_vis = False
        )

        vis.plot()
    
    def test_sim(self):
        """
        Runs the simulation using more visually appealing values
        """
        xray = Xray(
            FWHM = 10,
            rotation = 0
        )

        gamma = Gamma(
            x_pos = -300,
            pulse_length = 100,
            height = 50,
            off_axis_dist = 100
        )

        vis = Visualiser(
            xray_bath = xray,
            gamma_pulse = gamma,
            bath_vis = True
        )

        vis.plot()


if __name__ == '__main__':
    test = Test()
    test.test_sim()