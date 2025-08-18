"""
Defines the class Gamma, representing a gamma pulse in the
simulation.
This class provides geometric properties and methods to change
its dimensions and position.
Also includes matplotlib integration through patches.
"""
from matplotlib.patches import Rectangle


class Gamma:
    """
    Gamma pulse parameters and axes object

    Attributes:
        x_pos (float): x-coordinate of gamma pulse (left edge of rectangle) (mm)
        pulse_length (float): length of gamma pulse (mm)
        height (float): height of gamma pulse in 2D, for an azimuthal angle of 0, 
                    this corresponds to the radius (mm)
        off_axis_dist (float): perpendicular distance from bottom of gamma pulse 
                            to x-ray source when directly over head (mm)
        facecolor (tuple, optional): face colour of axes object. Defaults to (0, 0, 1, 0.5).
        edgecolor (tuple, optional): edge colour of axes object. Defaults to (1, 0, 0).
        linewidth (int, optional): line width of axes object. Defaults to 2.
    """
    def __init__(
            self,
            x_pos: float,
            pulse_length: float,
            height: float,
            off_axis_dist : float,
            facecolor = (0, 0, 1, 0.5),
            edgecolor = (1, 0, 0),
            linewidth = 2
        ):
        self.x_pos = x_pos
        self.pulse_length = pulse_length
        self.height = height
        self.off_axis_dist = off_axis_dist

        # matplotlib rectangle object for visualisation
        self.gamma_axes_obj = Rectangle(
            xy = [ x_pos, off_axis_dist ] ,
              width = pulse_length ,
              height = height ,
              facecolor = facecolor ,
              edgecolor = edgecolor ,
              linewidth = linewidth ,
              label = 'gamma pulse')

    ############ METHODS ###########################################################################
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

    def move(self, dx: float):
        """Moves gamma pulse object

        Args:
            dx (float): distance to move (mm)
        """
        self.gamma_axes_obj.set_x(self.get_x_pos() + dx)

    def set_x_pos(self, new_x_pos: float):
        """Sets the gamma pulse to a new defined x coordinate

        Args:
            new_x_pos (float): x coordinate to move to
        """
        self.x_pos = new_x_pos
        self.gamma_axes_obj.set_x(new_x_pos)

    def set_height(self, new_height: float):
        """change the height of the gamma pulse

        Args:
            new_height (float): new height of the gamma pulse
        """
        self.height = new_height

    ############ ACCESS METHODS ####################################################################
    def get_x_pos(self) -> float:
        """Access method for initial x position

        Returns:
            float: initial x-coordinate (mm)
        """
        return self.x_pos

    def get_pulse_length(self) -> float:
        """Access method for pulse length

        Returns:
            float: length of pulse (mm)
        """
        return self.pulse_length

    def get_height(self) -> float:
        """Access method for pulse height

        Returns:
            float: pulse height (mm)
        """
        return self.height

    def get_off_axis_dist(self) -> float:
        """Access method for the axial displacement

        Returns:
            float: axial displacement (mm)
        """
        return self.off_axis_dist

    def get_gamma_axes_obj(self) -> float:
        """Access method for the gamma axes object

        Returns:
            matplotlib.patches.Rectangle: Axes object for gamma pulse
        """
        return self.gamma_axes_obj
