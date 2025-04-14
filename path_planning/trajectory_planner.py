import rclpy
import numpy as np
from rclpy.node import Node
from queue import PriorityQueue
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory
from .spline_path import spline
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
import cv2

class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")
        self.declare_parameter('use_sampling', False)  # Parameter to choose between A* and RRT
        self.declare_parameter('use_spline', False)  # Parameter to spline the path
        self.declare_parameter('dilation_filter_size', 10)

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value
        self.use_sampling = self.get_parameter('use_sampling').get_parameter_value().bool_value
        self.use_spline = self.get_parameter('use_spline').get_parameter_value().bool_value
        self.dilation_filter_size = self.get_parameter('dilation_filter_size').get_parameter_value().integer_value

        # Create a QoS profile with transient local durability for the map subscription
        map_qos = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=1
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            qos_profile=map_qos)

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
        self.splined_trajectory = LineTrajectory(node=self, viz_namespace="/splined_trajectory") # Uncomment for visualization of the splined trajectory
        
        # Initialize map and pose variables
        self.map_info = None
        self.current_grid_pose = None
        self.current_world_pose = None
        
        # Debug: Print all parameters
        self.get_logger().info(f"Initialized with parameters: odom_topic={self.odom_topic}, map_topic={self.map_topic}, initial_pose_topic={self.initial_pose_topic}, use_sampling={self.use_sampling}, use_spline={self.use_spline}")
        self.get_logger().info(f"Visualization namespace: {self.trajectory.viz_namespace}")
        self.get_logger().info(f"Splined Trajectory namespace: {self.splined_trajectory.viz_namespace}")
        
        # Try to get the map from the topic directly
        self.get_logger().info(f"Waiting for map on topic: {self.map_topic}")

    def map_cb(self, msg):
        """Callback for receiving map updates."""
        self.get_logger().info(f"Map received: width={msg.info.width}, height={msg.info.height}, resolution={msg.info.resolution}")
        self.get_logger().info(f"Map origin: x={msg.info.origin.position.x}, y={msg.info.origin.position.y}")

        map_data = np.array(msg.data, dtype = np.int8).reshape((msg.info.height, msg.info.width))

        # converting between numpy and occupancy graph
        normalized_map = np.where(map_data == 0, 0, 1).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.dilation_filter_size, self.dilation_filter_size))
        dilated_normalized_map = cv2.dilate(normalized_map, kernel)
        dilated_map = np.where(dilated_normalized_map == 1, 100, 0).astype(np.int8)

        dilated_msg = OccupancyGrid()
        dilated_msg.header = msg.header
        dilated_msg.info = msg.info
        dilated_msg.data = dilated_map.flatten().tolist()

        self.map_info = dilated_msg
        # self.map_info = msg # Comment when using the dilated map
        self.get_logger().info("Map received and dilated")

    def pose_cb(self, msg):
        """Callback for receiving current pose updates."""
        self.current_world_pose = msg.pose.pose
        world_x = msg.pose.pose.position.x
        world_y = msg.pose.pose.position.y
        self.get_logger().info(f"Pose received: x={world_x}, y={world_y}")
        
        if self.map_info is not None:
            self.current_grid_pose = self.world_to_grid(world_x, world_y)
            self.get_logger().info(f"Current grid pose: {self.current_grid_pose}")

    def goal_cb(self, msg):
        """Callback for receiving goal pose updates."""
        if not self.map_info or not self.current_grid_pose:
            self.get_logger().warn('Map data or current pose not available, cannot plan path.')
            return

        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y
        self.get_logger().info(f"Goal received: x={goal_x}, y={goal_y}")
        
        goal_grid = self.world_to_grid(goal_x, goal_y)
        self.get_logger().info(f"Goal grid pose: {goal_grid}")
        
        self.plan_path(self.current_grid_pose, goal_grid)

    def plan_path(self, start, goal):
        """Plans the path from start to goal using A* algorithm and publishes it."""
        self.get_logger().info("Starting path planning...")
        
        if self.use_sampling:
            # RRT implementation (not shown here)
            self.get_logger().info("Using RRT algorithm")
            path = self.rrt(start, goal)
        else:
            # A* implementation
            self.get_logger().info("Using A* algorithm")
            path = self.a_star(start, goal)
        
        if path:
            self.get_logger().info(f"Path found with {len(path)} points")
            # Convert path to world coordinates and create trajectory
            self.trajectory.clear()
            
            for grid_cell in path:
                world_pos = self.grid_to_world(grid_cell[0], grid_cell[1])
                # The addPoint method takes a single tuple parameter
                self.trajectory.addPoint(world_pos)

            # Spline the trajectory for smoother trajectory
            if self.use_spline:
                self.get_logger().info("Applying spline to trajectory")
                self.trajectory.points = spline(self.trajectory.points)
            
            # Create and publish pose array
            pose_array = self.trajectory.toPoseArray()
            self.get_logger().info(f"Publishing trajectory with {len(pose_array.poses)} poses")
            self.traj_pub.publish(pose_array)
            
            # Publish visualization
            self.get_logger().info("Publishing visualization")
            self.trajectory.publish_viz()

            # Publish splined visualization to compare with original trajectory (when use_spline is False, as otherwise trajectory is the splined trajectory)
            if not self.use_spline:
                self.get_logger().info("Publishing splined visualization")
                self.splined_trajectory.clear()
                self.splined_trajectory.points = spline(self.trajectory.points)
                self.splined_trajectory.publish_trajectory(color=(0.0, 1.0, 1.0, 0.7))
            
            # The trajectory already has the start and end points from the path
            # No need to publish them separately
        else:
            self.get_logger().warn("No path found, trajectory will be empty")
            self.trajectory.clear()
            pose_array = self.trajectory.toPoseArray()
            self.traj_pub.publish(pose_array)
            self.trajectory.publish_viz()
        
        self.get_logger().info("Path planning completed.")

    def a_star(self, start, goal):
        """Implements the A* pathfinding algorithm."""
        self.get_logger().info(f"A* start grid: {start}")
        self.get_logger().info(f"A* goal grid: {goal}")
        
        # Skip walkable checks - just proceed with the algorithm
        frontier = PriorityQueue()
        frontier.put((0, start))
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0
        
        nodes_explored = 0
        max_nodes = 10000000  # Limit to prevent infinite loops
        
        while not frontier.empty() and nodes_explored < max_nodes:
            current = frontier.get()[1]
            nodes_explored += 1
            
            # Log progress every 1000 nodes
            if nodes_explored % 1000 == 0:
                self.get_logger().info(f"A* explored {nodes_explored} nodes so far")
            
            if current == goal:
                self.get_logger().info(f"A* found a path after exploring {nodes_explored} nodes")
                break
            
            for next_node in self.get_neighbors(current):
                new_cost = cost_so_far[current] + self.cost(current, next_node)
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    # Use a tie-breaking heuristic that slightly favors paths toward the goal
                    # This helps prevent zigzagging and encourages more direct paths
                    priority = new_cost + self.heuristic(goal, next_node) * 1.001
                    frontier.put((priority, next_node))
                    came_from[next_node] = current
        
        if goal not in came_from:
            self.get_logger().warn(f"No path found after exploring {nodes_explored} nodes")
            
            # Try to find the closest point to the goal that we did reach
            if came_from:
                closest_node = min(came_from.keys(), key=lambda n: self.heuristic(n, goal))
                self.get_logger().info(f"Returning path to closest reachable point: {closest_node}")
                path = self.reconstruct_path(came_from, start, closest_node)
                # Add the original goal at the end if possible
                if self.is_edge_free(closest_node, goal):
                    path.append(goal)
                return path
            
            # If all else fails, return direct path
            return [start, goal]
        
        return self.reconstruct_path(came_from, start, goal)

    def get_neighbors(self, node):
        """Returns the neighboring nodes, checking for collisions."""
        # Include all 8 directions for better path flexibility
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
        result = []
        
        for dx, dy in directions:
            neighbor = (node[0] + dx, node[1] + dy)
            
            # Check if the neighbor cell is free
            if self.is_cell_free(neighbor[0], neighbor[1]):
                # For diagonal moves, also check the adjacent cells to ensure we can actually move there
                if abs(dx) == 1 and abs(dy) == 1:
                    # Check if we can move horizontally and vertically to reach the diagonal
                    if (self.is_cell_free(node[0] + dx, node[1]) and 
                        self.is_cell_free(node[0], node[1] + dy)):
                        result.append(neighbor)
                else:
                    result.append(neighbor)
        
        # Debug logging
        if not result:
            self.get_logger().debug(f"No valid neighbors found for node {node}")
        
        return result

    def cost(self, current, next_node):
        """
        Calculate the cost of moving from current to next_node.
        Diagonal moves cost more than orthogonal moves.
        """
        dx = abs(next_node[0] - current[0])
        dy = abs(next_node[1] - current[1])
        
        # Diagonal movement costs sqrt(2) ≈ 1.414
        if dx == 1 and dy == 1:
            return 1.414
        else:
            return 1.0

    def heuristic(self, a, b):
        """
        Calculate the heuristic value (Euclidean distance).
        This is an admissible heuristic for grid-based movement.
        """
        # Euclidean distance
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def reconstruct_path(self, came_from, start, goal):
        """Reconstructs the path from start to goal using the came_from map."""
        current = goal
        path = []
        
        while current != start:
            path.append(current)
            current = came_from[current]
        
        path.append(start)
        path.reverse()
        
        return path

    def is_cell_free(self, grid_x, grid_y):
        """Check if a specific grid cell is free (not an obstacle)."""
        if not self.map_info:
            return False
            
        width = self.map_info.info.width
        height = self.map_info.info.height
        
        # Handle negative coordinates and ensure they're within bounds
        grid_x = (grid_x + width) % width
        grid_y = (grid_y + height) % height
        
        # Check bounds
        if grid_x < 0 or grid_y < 0 or grid_x >= width or grid_y >= height:
            return False
            
        # Calculate index
        idx = grid_y * width + grid_x
        
        # Check if index is valid
        if idx < 0 or idx >= len(self.map_info.data):
            return False
            
        # Check if cell is traversable (not an obstacle)
        return self.map_info.data[idx] == 0

    def is_edge_free(self, node1, node2):
        """
        Checks if the edge between two grid cells is collision-free
        using linear interpolation and sampling.
        """
        # If nodes are the same, edge is free
        if node1 == node2:
            return True
            
        # For adjacent cells, just check if both endpoints are free
        if abs(node1[0] - node2[0]) <= 1 and abs(node1[1] - node2[1]) <= 1:
            return self.is_cell_free(node1[0], node1[1]) and self.is_cell_free(node2[0], node2[1])
        
        # For longer edges, sample points along the edge
        x1, y1 = node1
        x2, y2 = node2
        
        # Calculate distance in grid cells
        dist = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        
        # Number of points to check along the edge (at least 3)
        steps = max(int(dist * 2), 3)
        
        # Check points along the edge
        for i in range(steps + 1):
            # Linear interpolation
            x = x1 + (x2 - x1) * i / steps
            y = y1 + (y2 - y1) * i / steps
            
            # Convert to integer grid coordinates
            grid_x = int(round(x))
            grid_y = int(round(y))
            
            # Check if this point is free
            if not self.is_cell_free(grid_x, grid_y):
                return False
                
        return True

    def is_in_bounds(self, node):
        """Checks if the node is within the map bounds and not in collision."""
        return self.is_cell_free(node[0], node[1])

    def world_to_grid(self, world_x, world_y):
        """Converts world coordinates to grid coordinates."""
        if not self.map_info:
            return None
        
        origin_x = self.map_info.info.origin.position.x
        origin_y = self.map_info.info.origin.position.y
        resolution = self.map_info.info.resolution
        
        grid_x = - int((world_x - origin_x) / resolution)
        grid_y = - int((world_y - origin_y) / resolution)
        
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        """Converts grid coordinates to world coordinates."""
        if not self.map_info:
            return None
        
        origin_x = self.map_info.info.origin.position.x
        origin_y = self.map_info.info.origin.position.y
        resolution = self.map_info.info.resolution
        
        world_x = - grid_x * resolution + origin_x
        world_y = - grid_y * resolution + origin_y
        
        return world_x, world_y

    def rrt(self, start, goal):
        """Implements the RRT pathfinding algorithm."""
        # RRT implementation (placeholder)
        self.get_logger().warn("RRT not implemented yet")
        return None


def main(args=None):
    rclpy.init(args=args)
    node = PathPlan()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
