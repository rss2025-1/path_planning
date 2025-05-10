import rclpy
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
from visualization_msgs.msg import Marker
from .rviz_tools import RVizTools
from std_msgs.msg import Bool

from .utils import LineTrajectory
import time

class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """
    
    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")
        self.declare_parameter('trajectory', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value
   
        self.lookahead = 1.25 #0.5  # FILL IN #
        self.speed = 1.0 #0.5  # FILL IN #
        self.wheelbase_length = 0.34  # FILL IN #

        self.trajectory = LineTrajectory(self, "/followed_trajectory")

        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               10)
        self.odom_sub = self.create_subscription(Odometry,
                                                self.odom_topic,
                                                self.pose_callback,
                                                10)
        self.radius_pub = self.create_publisher(Marker, "/radius", 10)
        self.line_pub = self.create_publisher(Marker, "/line", 10)
        self.closest_segment_pub = self.create_publisher(Marker, "/closest_segment", 10)

        self.log_counter = 0
        self.goal_pose = None
        self.goal_reached_threshold = 0.2 # meters, tune as needed

        self.goal_reached_pub = self.create_publisher(Bool, "/goal_reached", 10)

    def pose_callback(self, odometry_msg):
        car_x, car_y, yaw = self.get_vehicle_pose(odometry_msg)

        # Check if goal is reached
        if self.goal_pose is not None:
            current_pos = np.array([car_x, car_y])
            dist_to_goal = np.linalg.norm(current_pos - self.goal_pose)
            if dist_to_goal < self.goal_reached_threshold:
                self.get_logger().info(f"Goal reached (distance: {dist_to_goal:.2f}m < {self.goal_reached_threshold}m). Stopping.")
                self.goal_reached_pub.publish(Bool(data=bool(True)))
                self.goal_pose = None
                start = time.time()
                while time.time() - start < 5.0:
                    drive_msg = AckermannDriveStamped()
                    drive_msg.header.stamp = self.get_clock().now().to_msg()
                    drive_msg.drive.speed = -1.0
                    self.drive_pub.publish(drive_msg)
                return 

        # If no goal is set, stop the car; this occurs at the start and when a goal is cleared
        if self.goal_pose is None:
            self.publish_drive_command(steering_angle=0.0, speed=0.0)
            return

        lookahead_point = self.find_lookahead_point(car_x, car_y)

        if lookahead_point is None:
            if self.log_counter % 125 == 0:
                self.get_logger().info("No lookahead point found. Stopping.")
            self.log_counter += 1
            return
        
        steering_angle = self.compute_steering_angle(car_x, car_y, yaw, lookahead_point)
        self.publish_drive_command(steering_angle)
        self.get_logger().info(f"steering_angle: {steering_angle}")
        self.log_counter += 1

    def find_closest_point_on_trajectory(self, traj, P):
        """
        Find the closest point on the trajectory to the car.
        Returns:
            closest_point (np.array): Closest point on the trajectory
            closest_idx (int): Index of the starting point of the segment containing the closest point
        """
        A = traj[:-1]
        B = traj[1:]

        AB = B - A  # Vector from A to B for all segments, shape (N-1, 2)
        AP = P - A  # Vector from A to P for all segments, shape (N-1, 2)

        t = np.sum(AP * AB, axis=1) / np.sum(AB * AB, axis=1)
        t_clamped = np.clip(t, 0, 1)  # Clamping t to [0, 1]
        proj = A + t_clamped[:, np.newaxis] * AB  # Projected points on the segments
        dists = np.linalg.norm(P - proj, axis=1)
        closest_idx = np.argmin(dists)

        return closest_idx

    def find_lookahead_point(self, car_x, car_y):
        """
        Use pure pursuit strategy:
        - Find the point on the trajectory nearest to the car.
        - Then search from that point forward for a circle-line intersection
        with radius = lookahead distance.
        """

        traj = np.array(self.trajectory.points)
        
        P = np.array([car_x, car_y])
        if traj.shape[0] < 2:
            #self.get_logger().error("Not enough points in trajectory to compute closest point.")
            return None

        # Step 1: Find closest point on trajectory
        closest_idx = self.find_closest_point_on_trajectory(traj, P)

        # Step 2: Search for intersection with lookahead circle
        for i in range(closest_idx, len(traj) - 1):
            A =traj[i]
            B = traj[i + 1]

            # If the furthest endpoint of the segment is inside the lookahead circle, skip this segment.
            if np.linalg.norm(B - P) < self.lookahead:
                continue

            d = B - A
            f = A - P

            a = np.dot(d, d)
            b = 2 * np.dot(f, d)
            c = np.dot(f, f) - self.lookahead**2

            discriminant = b**2 - 4*a*c
            if discriminant < 0:
                continue  # No intersection

            sqrt_disc = np.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / (2 * a)
            t2 = (-b + sqrt_disc) / (2 * a)
            
            return (A + max(t1,t2)* d).tolist()

        return traj[closest_idx + 1]  # No intersection found, use the farther endpoint of the closest segment

    
    def get_vehicle_pose(self, odometry_msg):
        """ Extract vehicle position (x, y) and orientation (yaw) from the odometry message. """
        position = odometry_msg.pose.pose.position
        orientation = odometry_msg.pose.pose.orientation

        # Convert quaternion to yaw angle
        q = orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        return position.x, position.y, yaw
    
    def compute_steering_angle(self, car_x, car_y, yaw, lookahead_point):
        """ Compute the steering angle based on the pure pursuit algorithm. """
        dx = lookahead_point[0] - car_x
        dy = lookahead_point[1] - car_y

        # rotate lookahead point into the robot frame 
        local_x = np.cos(-yaw) * dx - np.sin(-yaw) * dy
        local_y = np.sin(-yaw) * dx + np.cos(-yaw) * dy
        # self.get_logger().info(f"(local_X, local_y): {local_x, local_y}")
        # self.get_logger().info(f"(global_X, global_y): {dx, dy}")
        # self.get_logger().info(f"lookahead pt: {lookahead_point}") 
        # self.get_logger().info(f"car: ({car_x}, {car_y}, {yaw})") 
        RVizTools.plot_circle(self.lookahead, self.radius_pub) #visualize pure pursuit
        RVizTools.plot_line(np.array([0,local_x]), np.array([0,local_y]), self.line_pub)
        if local_x <= 0:
            # Lookahead point is behind the vehicle
            # self.get_logger().info("Lookahead point is behind the vehicle.")
            return 0.0  # No steering needed if the point is behind

        # Compute the curvature and then the steering angle using the bicycle model
        eta = np.arctan2(local_y, local_x)
        steering_angle = np.arctan2(2 * self.wheelbase_length * np.sin(eta), self.lookahead)

        return steering_angle
    
    def publish_drive_command(self, steering_angle, speed=None):
        """ Publish the Ackermann drive message with the computed steering angle and speed. """
        if speed is None:
            speed = self.speed
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

    def trajectory_callback(self, msg):
        # self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")
        # self.get_logger().info(f"msg.poses type is {msg.poses}")
        
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        if len(msg.poses) > 0:
            last_pose = msg.poses[-1].position
            self.goal_pose = np.array([last_pose.x, last_pose.y])
            self.get_logger().info(f"New trajectory received. Goal set to: ({self.goal_pose[0]:.2f}, {self.goal_pose[1]:.2f})")
        else:
            self.goal_pose = None # Reset goal if trajectory is empty
            self.get_logger().info("Received empty trajectory. Goal cleared.")

        self.initialized_traj = True


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
