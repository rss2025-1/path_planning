import rclpy
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf_transformations import quaternion_from_euler, euler_from_quaternion


from .utils import LineTrajectory


class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.lookahead = 0.5  # FILL IN #
        self.speed = 0.5  # FILL IN #
        self.wheelbase_length = 0.34  # FILL IN #

        # self.trajectory = LineTrajectory("/followed_trajectory")
        self.trajectory = LineTrajectory("/example_trajectories/trajectory_2.traj")


        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)
        self.odom_sub = self.create_subscription(Odometry,
                                                self.odom_topic,
                                                self.pose_callback,
                                                1)
        self.log_counter = 0

    def pose_callback(self, odometry_msg):
        car_x, car_y, yaw = self.get_vehicle_pose(odometry_msg)

        lookahead_point = self.find_lookahead_point(car_x, car_y)

        if lookahead_point is None:
            if self.log_counter % 125 == 0:
                self.get_logger().info("No lookahead point found. Stopping.")
                # self.get_logger().info(f"{car_x, car_y, yaw}")
            self.log_counter += 1
            return

        steering_angle = self.compute_steering_angle(car_x, car_y, yaw, lookahead_point)
        self.publish_drive_command(steering_angle)

        # if self.log_counter % 125 == 0:
        #     self.get_logger().info(f"{car_x, car_y, yaw}")

        self.log_counter += 1

    def find_closest_point_on_trajectory(self, car_x, car_y):
        """
        Find the closest point on the trajectory to the car.
        Returns:
            closest_point (np.array): Closest point on the trajectory
            closest_idx (int): Index of the starting point of the segment containing the closest point
        """
        traj = np.array(self.trajectory.points)  # shape (N, 2)
    
        P = np.array([car_x, car_y])

        min_dist = float('inf')
        closest_point = None
        closest_idx = 0
        self.get_logger().info(f"{self.trajectory.points}")
        for i in range(len(traj) - 1):
            A = traj[i]
            B = traj[i + 1]
            AB = B - A
            AP = P - A
            self.get_logger().info(f"{i, AB, AP}")
            t = np.dot(AP, AB) / np.dot(AB, AB)
            t_clamped = np.clip(t, 0, 1)
            proj = A + t_clamped * AB

            dist = np.linalg.norm(P - proj)
            if dist < min_dist:
                min_dist = dist
                closest_point = proj
                closest_idx = i

        return closest_point, closest_idx

    def find_lookahead_point(self, car_x, car_y):
        """
        Use pure pursuit strategy:
        - Find the point on the trajectory nearest to the car.
        - Then search from that point forward for a circle-line intersection
        with radius = lookahead distance.
        """

        # Step 1: Find closest point on trajectory
        closest_point, closest_idx = self.find_closest_point_on_trajectory(car_x, car_y)

        # Step 2: Search for intersection with lookahead circle
        traj = np.array(self.trajectory.points)
        P = np.array([car_x, car_y])

        for i in range(closest_idx, len(traj) - 1):
            A = traj[i]
            B = traj[i + 1]
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
            # for t in [t1, t2]:
            #     if 0 <= t <= 1:
            #         intersection = A + t * d
            #         return intersection.tolist()

        return None  # No intersection found

    
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

        if local_x <= 0:
            # Lookahead point is behind the vehicle
            self.get_logger().info("Lookahead point is behind the vehicle.")
            return 0  # No steering needed if the point is behind

        # Compute the curvature and then the steering angle using the bicycle model
        eta = np.arctan2(local_y, local_x)
        R = self.lookahead / (2 * np.sin(eta))
        steering_angle = np.arctan2(self.wheelbase_length, R)

        return steering_angle
    def publish_drive_command(self, steering_angle):
        """ Publish the Ackermann drive message with the computed steering angle and speed. """
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = self.speed
        self.drive_pub.publish(drive_msg)
    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
