import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory


class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.pose_cb,
            10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

    def map_cb(self, msg):
        self.map = msg
        self.get_logger().info("Map received")

    def pose_cb(self, pose):
        self.start_pose = pose.pose
        self.get_logger().info("Pose received")

    def goal_cb(self, msg):
        self.goal_pose = msg.pose
        self.get_logger().info("Goal received")
        if hasattr(self, 'start_pose') and hasattr(self, 'map'):
            self.plan_path(self.start_pose, self.goal_pose, self.map)
        else:
            self.get_logger().warn("Waiting for start pose or map.")

    def plan_path(self, start_point, end_point, map):
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()

        # ===== HELPER FUNCTIONS =====
        def world_to_grid(x, y, map_msg):
            resolution = map_msg.info.resolution
            origin = map_msg.info.origin.position
            u = int((x - origin.x) / resolution)
            v = int((y - origin.y) / resolution)
            return u, v

        def is_free(x, y, map_msg):
            u, v = world_to_grid(x, y, map_msg)
            width = map_msg.info.width
            height = map_msg.info.height
            idx = v * width + u
            if u < 0 or v < 0 or u >= width or v >= height:
                return False
            return map_msg.data[idx] == 0  # 0 = free space

        def euclidean_distance(p1, p2):
            return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


        def a_star(start_point, end_point, map):
            # Implement A* algorithm here
            pass

        def rrt(start_point, end_point, map):
            # Implement RRT algorithm here
            pass


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
