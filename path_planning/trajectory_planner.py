import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from skimage.morphology import dilation, square
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory
import numpy as np
import heapq
import math


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

        map_data = np.array(msg.data, dtype = np.int8).reshape((msg.info.height, msg.info.width))

        # converting between numpy and occupancy graph
        normalized_map = np.where(map_data > 0, 1, 0).astype(np.uint8)
        dilated_normalized_map = dilation(normalized_map, square(10))
        dilated_map = np.where(dilated_normalized_map == 1, 100, 0).astype(np.int8)
        dilated_map[map_data == -1] = -1

        dilated_msg = OccupancyGrid()
        dilated_msg.header = msg.header
        dilated_msg.info = msg.info
        dilated_msg.data = dilated_map.flatten().tolis()

        self.map_dilated = dilated_msg
        self.get_logger().info("Map received and dilated")

    def pose_cb(self, pose):
        self.start_pose = pose.pose
        self.get_logger().info("Pose received")

    def goal_cb(self, msg):
        self.goal_pose = msg.pose
        self.get_logger().info("Goal received")
        if hasattr(self, 'start_pose') and hasattr(self, 'map'):
            self.plan_path(self.start_pose, self.goal_pose, self.map_dilated)
        else:
            self.get_logger().warn("Waiting for start pose or map.")

    def plan_path(self, start_point, end_point, map):
        # Clear any existing trajectory
        self.trajectory.clear()
        
        # Get path using A*
        path = a_star(start_point, end_point, map)
        # path = dfs_search(start_point, end_point, map)  # <-- get path with dfs
        
        # If path was found, add points to trajectory and publish
        if path and len(path) > 0:
            for point in path:
                self.trajectory.addPoint(point)
            
        # Publish trajectory regardless (empty if no path found)
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()

        # ===== HELPER FUNCTIONS =====
        def world_to_grid(x, y, map_msg):
            resolution = map_msg.info.resolution
            origin = map_msg.info.origin.position
            u = int((x - origin.x) / resolution)
            v = int((y - origin.y) / resolution)
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
            return map_msg.data[idx] == 0  # 0 = free space
        
        def is_edge_free(p1, p2, map_msg, steps=10):
            for i in range(steps + 1):
                u = p1[0] + (p2[0] - p1[0]) * i / steps
                v = p1[1] + (p2[1] - p1[1]) * i / steps
                if not is_free(u, v, map_msg):
                    return False
            return True

        def euclidean_distance(p1, p2):
            return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

        def a_star(start_point, end_point, map_msg):
            """
            A* algorithm implementation.
            
            Args:
                start_point: PoseWithCovariance with start position
                end_point: Pose with goal position
                map_msg: The occupancy grid map
                
            Returns:
                List of (x, y) tuples representing the path in world coordinates
            """
            # Convert start and goal to grid coordinates
            start_x, start_y = start_point.position.x, start_point.position.y
            goal_x, goal_y = end_point.position.x, end_point.position.y
            
            start_grid = world_to_grid(start_x, start_y, map_msg)
            goal_grid = world_to_grid(goal_x, goal_y, map_msg)
            
            # Check if start or goal is in an obstacle
            if not is_free(start_x, start_y, map_msg):
                self.get_logger().warn("Start position is in an obstacle!")
                return []
                
            if not is_free(goal_x, goal_y, map_msg):
                self.get_logger().warn("Goal position is in an obstacle!")
                return []
            
            # Initialize the open and closed sets
            open_set = []
            heapq.heappush(open_set, (0, start_grid))
            came_from = {start_grid: None}
            cost_so_far = {start_grid: 0}
            
            # Define possible movements (8-connected grid)
            movements = [
                (1, 0), (0, 1), (-1, 0), (0, -1),  # 4-connected
                (1, 1), (-1, 1), (-1, -1), (1, -1)  # Diagonals
            ]
            
            while open_set:
                # Get the node with the lowest f-score
                _, current = heapq.heappop(open_set)
                
                # If we've reached the goal, reconstruct and return the path
                if current == goal_grid:
                    # Reconstruct path in grid coordinates
                    grid_path = []
                    while current != start_grid:
                        grid_path.append(current)
                        current = came_from[current]
                    grid_path.append(start_grid)
                    grid_path.reverse()
                    
                    # Convert path to world coordinates
                    world_path = []
                    for grid_x, grid_y in grid_path:
                        world_x, world_y = grid_to_world(grid_x, grid_y, map_msg)
                        world_path.append((world_x, world_y))
                    
                    return world_path
                    
                # Explore neighbors
                for dx, dy in movements:
                    next_x, next_y = current[0] + dx, current[1] + dy
                    next_node = (next_x, next_y)
                    
                    # Check if the next node is valid (within map and not an obstacle)
                    next_world_x, next_world_y = grid_to_world(next_x, next_y, map_msg)
                    if is_free(next_world_x, next_world_y, map_msg):
                        # Calculate movement cost (diagonal movements cost more)
                        movement_cost = 1.0 if dx*dy == 0 else 1.414
                        
                        # Calculate new cost
                        new_cost = cost_so_far[current] + movement_cost
                        
                        # If we haven't visited this node or found a better path
                        if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                            cost_so_far[next_node] = new_cost
                            # Use Euclidean distance as heuristic
                            heuristic = math.sqrt((goal_grid[0] - next_x)**2 + (goal_grid[1] - next_y)**2)
                            priority = new_cost + heuristic
                            heapq.heappush(open_set, (priority, next_node))
                            came_from[next_node] = current
            
            # If we get here, no path was found
            self.get_logger().warn("No path found!")
            return []
        
        def steer(from_point, to_point, max_dist=0.5):
            # Move from "from_point" toward "to_point" but limit step size
            theta = math.atan2(to_point[1] - from_point[1], to_point[0] - from_point[0])
            dist = min(max_dist, euclidean_distance(from_point, to_point))
            new_point = (from_point[0] + dist * math.cos(theta), from_point[1] + dist * math.sin(theta))
            return new_point

        def rrt(start_point, end_point, map):
            start = (start_point.position.x, start_point.position.y)
            goal = (end_point.position.x, end_point.position.y)

            tree = {start: None}  # parent dictionary: node -> parent
            nodes = [start]

            max_iters = 5000
            goal_threshold = 0.5  # meters

            map_bounds = [
                (map.info.origin.position.x, map.info.origin.position.x + map.info.width * map.info.resolution),
                (map.info.origin.position.y, map.info.origin.position.y + map.info.height * map.info.resolution)
            ]

            for _ in range(max_iters):

                if random.random() < 0.1: # 10% chance to sample the goal
                    sample = goal
                else:
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

                for (x,y) in path:
                    self.trajectory.addPoint(x,y,0.0)

                self.traj_pub.publish(self.trajectory.toPoseArray())
                self.trajectory.publish_viz()


        def dfs_search(start_point, end_point, map):
            start_x, start_y = start_point.position.x, start_point.position.y
            goal_x, goal_y = end_point.position.x, end_point.position.y

            start_grid = world_to_grid(start_x, start_y, map)
            goal_grid = world_to_grid(goal_x, goal_y, map)

            if not is_free(start_x, start_y, map):
                self.get_logger().warn("Start position not free")
                return []

            if not is_free(goal_x, goal_y, map):
                self.get_logger().warn("Goal position not free")
                return []

            stack = [start_grid]
            came_from = {start_grid: None}
            visited = set()

            # 4 connected grid
            movements = [(1, 0), (0, 1), (-1, 0), (0, -1)]

            while stack:
                current = stack.pop()

                if current == goal_grid:
                    grid_path = []
                    while current is not None:
                        grid_path.append(current)
                        current = came_from[current]
                    grid_path.reverse()

                    world_path = [grid_to_world(x, y, map) for x, y in grid_path]
                    return world_path

                visited.add(current)

                for dx, dy in movements:
                    next_node = (current[0] + dx, current[1] + dy)
                    next_world = grid_to_world(next_node[0], next_node[1], map)

                    if next_node not in visited and is_free(*next_world, map):
                        came_from[next_node] = current
                        stack.append(next_node)
            return []



def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
