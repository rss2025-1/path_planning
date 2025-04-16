from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
import numpy as np
class RVizTools:
    @staticmethod
    def plot_line(x, y, publisher, color = (1., 0., 0.), frame = "/base_link"):
        """
        Publishes the points (x, y) to publisher
        so they can be visualized in rviz as
        connected line segments.
        Args:
            x, y: The x and y values. These arrays
            must be of the same length.
            publisher: the publisher to publish to. The
            publisher must be of type Marker from the
            visualization_msgs.msg class.
            color: the RGB color of the plot.
            frame: the transformation frame to plot in.
        """
        # Construct a line
        line_seg = Marker()
        line_seg.type = Marker.LINE_STRIP
        line_seg.header.frame_id = frame

        # Set the size and color
        line_seg.scale.x, line_seg.scale.y = 0.1, 0.1

        line_seg.color.a, line_seg.color.r, line_seg.color.g, line_seg.color.b = 1., color[0], color[1], color[2]

        # Fill the line with the desired values
        for xi, yi in zip(x, y):
            p = Point()
            p.x, p.y = xi, yi
            line_seg.points.append(p)

        publisher.publish(line_seg)
    def plot_circle(radius,  publisher, color=(0., 1., 0.), frame="/base_link", num_points=100):
   
        theta = np.linspace(0, 2 * np.pi, num_points)
        x =  radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        # Close the circle by appending the starting point again
        # x = np.append(x, x[0])
        # y = np.append(y, y[0])
        
        RVizTools.plot_line(x, y, publisher, color=color, frame=frame)
