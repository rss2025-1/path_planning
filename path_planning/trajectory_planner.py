import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory
import numpy as np
import heapq
import math
import random
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


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

        # Visualization for RRT tree
        self.rrt_tree_pub = self.create_publisher(
            MarkerArray, 
            "/rrt_tree", 
            10
        )

        self.rrt_points_pub = self.create_publisher(
            Marker,
            "/rrt_points",
            10
        )


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

        # ===== HELPER FUNCTIONS =====
        def world_to_grid(x, y, map_msg):
            resolution = map_msg.info.resolution
            origin = map_msg.info.origin.position
            
            u = int((origin.x - x) / resolution)
            v = int((origin.y - y) / resolution)
            return u, v

        def grid_to_world(u, v, map_msg):
            resolution = map_msg.info.resolution
            origin = map_msg.info.origin.position
            x = u * resolution + origin.x
            y = v * resolution + origin.y
            return x, y

        def is_free(x, y, map_msg):
            u, v = world_to_grid(x, y, map_msg)
            width = map_msg.info.width
            height = map_msg.info.height
            idx = v * width + u
            if u < 0 or v < 0 or u >= width or v >= height:
                return False
            return map_msg.data[idx] == 0 
        
        def is_edge_free(p1, p2, map_msg, step_size=0.05):
            dist = euclidean_distance(p1, p2)
            steps = max(int(dist / step_size), 1)
            for i in range(steps + 1):
                u = p1[0] + (p2[0] - p1[0]) * i / steps
                v = p1[1] + (p2[1] - p1[1]) * i / steps
                if not is_free(u, v, map_msg):
                    return False
            return True

        def euclidean_distance(p1, p2):
            return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        
        def steer(from_point, to_point, max_dist=0.5):
            # Move from "from_point" toward "to_point" but limit step size
            theta = math.atan2(to_point[1] - from_point[1], to_point[0] - from_point[0])
            dist = min(max_dist, euclidean_distance(from_point, to_point))
            new_point = (from_point[0] + dist * math.cos(theta), from_point[1] + dist * math.sin(theta))
            return new_point
        
        def make_edge_marker(start, end, id):
            marker = Marker()
            marker.header.frame_id = "map"  # assuming your map frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "rrt_tree"
            marker.id = id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.02  # Line width
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            p_start = Point()
            p_start.x, p_start.y, p_start.z = start[0], start[1], 0.0
            p_end = Point()
            p_end.x, p_end.y, p_end.z = end[0], end[1], 0.0
            marker.points = [p_start, p_end]
            return marker
    

        def rrt(start_point, end_point, map):
            tree_markers = MarkerArray()
            start = (start_point.pose.position.x, start_point.pose.position.y)
            goal = (end_point.position.x, end_point.position.y)
            self.get_logger().info(f"Start: {start}, Goal: {goal}")

            tree = {start: None}  # parent dictionary: node -> parent
            nodes = [start]

            max_iters = 50000
            goal_threshold = 0.5  # meters

            # map_bounds = [
            #     (map.info.origin.position.x, map.info.origin.position.x + map.info.width * map.info.resolution),
            #     (map.info.origin.position.y, map.info.origin.position.y + map.info.height * map.info.resolution)
            # ]
            map_bounds = [
                (map.info.origin.position.x - map.info.width * map.info.resolution, map.info.origin.position.x),
                (map.info.origin.position.y - map.info.height * map.info.resolution, map.info.origin.position.y)
            ]

            self.get_logger().info(f"Map bounds: {map_bounds}")
            self.get_logger().info(f"Map size: {map.info.width} x {map.info.height}")
            self.get_logger().info(f"Map resolution: {map.info.resolution}")
            self.get_logger().info(f"Map origin: {map.info.origin.position.x}, {map.info.origin.position.y}")

            for _ in range(max_iters):

                if random.random() < 0.1: # 10% chance to sample the goal
                    sample = goal
                else:
                    assert map.info.width > 0 and map.info.height > 0
                    sample = (random.uniform(map_bounds[0][0], map_bounds[0][1]),
                            random.uniform(map_bounds[1][0], map_bounds[1][1]))
                              
                # Find the nearest node in the tree
                nearest = min(nodes, key=lambda node: euclidean_distance(node, sample))

                # Steer towards the sample
                new_node = steer(nearest, sample)

                if not is_free(new_node[0], new_node[1], map):
                    continue

                if not is_edge_free(nearest, new_node, map):
                    continue

                # Add node
                nodes.append(new_node)
                tree[new_node] = nearest
                tree_markers.markers.append(make_edge_marker(nearest, new_node, len(tree_markers.markers)))

                if _ % 50 == 0:  # Every 50 steps
                    self.rrt_tree_pub.publish(tree_markers)

                # Check if goal is reached
                if euclidean_distance(new_node, goal) < goal_threshold:
                    tree[goal] = new_node
                    self.get_logger().info("Goal reached")
                    break

                

                
            path = []
            current = goal
            while current is not None:
                path.append(current)
                current = tree.get(current)

            path.reverse()

            self.get_logger().info(f"Path, {path}")
            # Clear any existing trajectory
            self.trajectory.clear()
            for (x,y) in path:
                self.trajectory.addPoint((x,y,0.0))

            self.traj_pub.publish(self.trajectory.toPoseArray())
            self.trajectory.publish_viz()
    
        rrt(self.start_pose, self.goal_pose, self.map)

        self.get_logger().info("Path planning completed.")


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
