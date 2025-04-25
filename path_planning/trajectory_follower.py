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
        self.declare_parameter('trajectory', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.lookahead = 0.5  # FILL IN #
        self.speed = 1.0 #0.5  # FILL IN #
        self.wheelbase_length = 0.34  # FILL IN #

        self.trajectory = LineTrajectory(self, "/followed_trajectory")

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
        self.cross_track_errors = []
        self.car_xy_path = []         # List of (x, y) tuples for the car
        self.traj_xy_path = []        # List of (x, y) trajectory points (only stored once)
    def pose_callback(self, odometry_msg):
        car_x, car_y, yaw = self.get_vehicle_pose(odometry_msg)
        self.car_xy_path.append((car_x, car_y))

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
    def compute_cross_track_error(self, car_x, car_y):
        traj = np.array(self.trajectory.points)
        P = np.array([car_x, car_y])
        closest_idx = self.find_closest_point_on_trajectory(traj, P)
    
        A = traj[closest_idx]
        B = traj[closest_idx + 1]

        AB = B - A
        AP = P - A

        proj_len = np.dot(AP, AB) / np.linalg.norm(AB)
        proj_point = A + (proj_len / np.linalg.norm(AB)) * AB

        error_vec = P - proj_point
        cte = np.linalg.norm(error_vec)

        # Determine sign using cross product: positive if point is left of trajectory
        cross = np.cross(AB, AP)
        if cross < 0:
            cte *= -1
        return cte
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
            self.get_logger().error("Not enough points in trajectory to compute closest point.")
            return None

        # Step 1: Find closest point on trajectory
        closest_idx = self.find_closest_point_on_trajectory(traj, P)

        # Step 2: Search for intersection with lookahead circle

        # Check the 5 closest segments to find an intersection
        for i in range(closest_idx, min(closest_idx + 5, len(traj) - 1)):
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

        if local_x <= 0:
            # Lookahead point is behind the vehicle
            self.get_logger().info("Lookahead point is behind the vehicle.")
            return 0.0  # No steering needed if the point is behind

        # Compute the curvature and then the steering angle using the bicycle model
        eta = np.arctan2(local_y, local_x)
        steering_angle = np.arctan2(2 * self.wheelbase_length * np.sin(eta), self.lookahead)
        steering_angle_threshold = 1e-10  # Adjust this value as needed
        if abs(steering_angle) < steering_angle_threshold:
            steering_angle = steering_angle_threshold * np.sign(steering_angle)
        return steering_angle
    
    def publish_drive_command(self, steering_angle, speed=None):
        """ Publish the Ackermann drive message with the computed steering angle and speed. """
        
        speed = self.speed
        drive_msg = AckermannDriveStamped()
        
        drive_msg.drive.steering_angle = steering_angle
        self.get_logger().info(f"{steering_angle}")
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True
    def destroy_node(self):
        if self.cross_track_errors:
            avg_cte = sum(self.cross_track_errors) / len(self.cross_track_errors)
            self.get_logger().info(f"\n\n========== Run Complete ==========\n"
                                f"Average Cross Track Error: {avg_cte:.3f} meters\n"
                                f"Number of Samples: {len(self.cross_track_errors)}\n"
                                f"==================================\n")
        else:
            self.get_logger().info("No cross track error data collected.")
        # with open("car_xy_4_125.csv", "w", newline="") as carfile:
        #     writer = csv.writer(carfile)
        #     writer.writerow(["Car_X", "Car_Y"])
        #     for x, y in self.car_xy_path:
        #         writer.writerow([x, y])
        #         super().destroy_node()
        with open("trajectory_xy_1.csv", "w", newline="") as trajfile:
            writer = csv.writer(trajfile)
            writer.writerow(["Traj_X", "Traj_Y"])
            for x, y in self.traj_xy_path:
                writer.writerow([x, y])

def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()

    def shutdown_handler(sig, frame):
        follower.get_logger().info("Shutting down node...")
        follower.destroy_node()  # This will now call your overridden method
        rclpy.shutdown()

    signal.signal(signal.SIGINT, shutdown_handler)

    try:
        rclpy.spin(follower)
    except KeyboardInterrupt:
        shutdown_handler(None, None)
    # rclpy.spin(follower)
    # rclpy.shutdown()
