#!/usr/bin/env python3

import heapq
import math
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from wildfire_msgs.msg import FireGrid

FIRE_THRESHOLD = 0.1
SAFETY_BUFFER_CELLS = 1
REPLAN_INTERVAL_S = 2.0
# After reaching a scout-assigned attack point, step along the fire perimeter
PERIMETER_STEP_INTERVAL_S = 3.5
# Refill commands publish (0,0); margin only for float noise.
REFILL_GOAL_R2 = 0.01  # squared dist from origin — above this ⇒ firefighting goal

# 8-connected neighbours (dx, dy, cost)
_NEIGHBOURS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (1, 1, 1.414),
]


def _world_to_grid(wx, wy, grid_size, cell_size):
    half = grid_size * cell_size / 2.0
    gx = int((wx + half) / cell_size)
    gy = int((wy + half) / cell_size)
    return max(0, min(gx, grid_size - 1)), max(0, min(gy, grid_size - 1))


def _grid_to_world(gx, gy, grid_size, cell_size):
    wx = (gx - grid_size / 2.0 + 0.5) * cell_size
    wy = (gy - grid_size / 2.0 + 0.5) * cell_size
    return wx, wy


def _build_obstacle_set(intensity, grid_size):
    """Return set of (gx, gy) cells that are on fire or within the safety buffer."""
    fire_cells = set()
    for gy in range(grid_size):
        for gx in range(grid_size):
            if intensity[gy * grid_size + gx] >= FIRE_THRESHOLD:
                fire_cells.add((gx, gy))

    obstacles = set(fire_cells)
    for (fx, fy) in fire_cells:
        for dx in range(-SAFETY_BUFFER_CELLS, SAFETY_BUFFER_CELLS + 1):
            for dy in range(-SAFETY_BUFFER_CELLS, SAFETY_BUFFER_CELLS + 1):
                nx, ny = fx + dx, fy + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    obstacles.add((nx, ny))
    return obstacles, fire_cells


def _astar(start, goal, obstacles, grid_size):
    """A* on an 8-connected grid. Returns list of (gx, gy) or None."""
    if goal in obstacles:
        return None

    open_set = [(0.0, start)]
    g_score = {start: 0.0}
    came_from = {}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        cx, cy = current
        for dx, dy, cost in _NEIGHBOURS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < grid_size and 0 <= ny < grid_size):
                continue
            if (nx, ny) in obstacles:
                continue
            tentative = g_score[current] + cost
            if tentative < g_score.get((nx, ny), float('inf')):
                g_score[(nx, ny)] = tentative
                h = math.hypot(goal[0] - nx, goal[1] - ny)
                heapq.heappush(open_set, (tentative + h, (nx, ny)))
                came_from[(nx, ny)] = current

    return None


def _simplify_path(path):
    """Remove collinear intermediate waypoints."""
    if len(path) <= 2:
        return list(path)
    simplified = [path[0]]
    for i in range(1, len(path) - 1):
        dx1 = path[i][0] - path[i - 1][0]
        dy1 = path[i][1] - path[i - 1][1]
        dx2 = path[i + 1][0] - path[i][0]
        dy2 = path[i + 1][1] - path[i][1]
        if (dx1, dy1) != (dx2, dy2):
            simplified.append(path[i])
    simplified.append(path[-1])
    return simplified


def _nearest_safe_cell(target_gx, target_gy, fire_cells, obstacles, grid_size):
    """BFS outward from the target to find the nearest non-obstacle cell."""
    from collections import deque
    start = (target_gx, target_gy)
    if start not in obstacles:
        return start
    visited = {start}
    queue = deque([start])
    while queue:
        cx, cy = queue.popleft()
        for dx, dy, _ in _NEIGHBOURS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < grid_size and 0 <= ny < grid_size):
                continue
            if (nx, ny) in visited:
                continue
            visited.add((nx, ny))
            if (nx, ny) not in obstacles:
                return (nx, ny)
            queue.append((nx, ny))
    return None


class NavigationController(Node):
    def __init__(self):
        super().__init__('navigation_controller')

        self.robot_name = self.get_namespace().strip('/')
        if not self.robot_name:
            self.robot_name = 'firefighter_1'

        self._tag = f'[NAV:{self.robot_name}]'
        self._control_tick = 0
        self._odom_count = 0

        self.declare_parameter('linear_speed', 2.0)
        self.declare_parameter('angular_speed', 1.0)
        self.declare_parameter('position_tolerance', 0.3)

        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.position_tolerance = self.get_parameter('position_tolerance').value

        self.current_position = Point()
        self.current_yaw = 0.0
        self.target_position = None

        # Fire grid state
        self._fire_intensity = None
        self._grid_size = 50
        self._cell_size = 1.0

        # Waypoint path (list of world-coord (x, y) tuples)
        self._waypoints: list[tuple[float, float]] = []
        self._final_target = None
        self._last_plan_time = 0.0

        self._pending_perimeter_after_scout = False
        self._perimeter_follow_active = False
        self._last_perimeter_step_time = 0.0

        self.cmd_vel_pub = self.create_publisher(
            Twist, f'/{self.robot_name}/cmd_vel', 10)

        self.create_subscription(
            Odometry, f'/{self.robot_name}/odometry',
            self._odom_cb, 10)
        self.create_subscription(
            Point, f'/{self.robot_name}/target_position',
            self._target_cb, 10)
        self.create_subscription(
            FireGrid, '/fire_grid', self._fire_grid_cb, 10)

        self.create_timer(0.1, self._control_loop)

        self.get_logger().info(f'{self._tag} Initialized with A* pathfinding')

    def _odom_cb(self, msg):
        self.current_position = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
        self._odom_count += 1

    def _fire_grid_cb(self, msg: FireGrid):
        self._fire_intensity = list(msg.intensity)
        self._grid_size = msg.width
        self._cell_size = msg.cell_size

    def _target_cb(self, msg):
        self._perimeter_follow_active = False
        r2 = msg.x * msg.x + msg.y * msg.y
        self._pending_perimeter_after_scout = r2 > REFILL_GOAL_R2

        self.target_position = msg
        self._final_target = (msg.x, msg.y)
        self._plan_path()

    def _plan_path(self):
        """Compute an A* path from current position to the target, avoiding fire."""
        if self._final_target is None:
            return

        tx, ty = self._final_target
        n = self._grid_size
        cs = self._cell_size

        # Without fire data, fall back to direct navigation
        if self._fire_intensity is None:
            self._waypoints = [(tx, ty)]
            self._last_plan_time = time.monotonic()
            return

        obstacles, fire_cells = _build_obstacle_set(self._fire_intensity, n)

        start = _world_to_grid(
            self.current_position.x, self.current_position.y, n, cs)
        goal = _world_to_grid(tx, ty, n, cs)

        start_in_obs = start in obstacles
        if start_in_obs:
            obstacles.discard(start)

        if goal in obstacles:
            safe = _nearest_safe_cell(goal[0], goal[1], fire_cells, obstacles, n)
            if safe is None:
                self.get_logger().warn(
                    f'{self._tag} Target is in fire and no safe cell nearby — stopping')
                self._waypoints = []
                self.target_position = None
                self._final_target = None
                self._last_plan_time = time.monotonic()
                return
            self.get_logger().info(
                f'{self._tag} Target in fire zone — redirecting to safe cell '
                f'({safe[0]}, {safe[1]})')
            goal = safe

        path = _astar(start, goal, obstacles, n)

        if start_in_obs:
            obstacles.add(start)

        if path is None:
            self.get_logger().warn(
                f'{self._tag} No path found — stopping')
            self._waypoints = []
            self.target_position = None
            self._final_target = None
            self._last_plan_time = time.monotonic()
            return

        path = _simplify_path(path)

        self._waypoints = [
            _grid_to_world(gx, gy, n, cs) for gx, gy in path[1:]
        ]
        self._last_plan_time = time.monotonic()
        self.get_logger().info(
            f'{self._tag} Planned path with {len(self._waypoints)} waypoints')

    def _grid_has_fire(self) -> bool:
        if self._fire_intensity is None:
            return False
        return any(v >= FIRE_THRESHOLD for v in self._fire_intensity)

    def _perimeter_world_points(self) -> list[tuple[float, float]]:
        """Non-fire cells that are 4-neighbours of a burning cell (world x, y)."""
        if self._fire_intensity is None:
            return []
        n = self._grid_size
        cs = self._cell_size
        intensity = self._fire_intensity
        perimeter_cells: set[tuple[int, int]] = set()
        for gy in range(n):
            for gx in range(n):
                if intensity[gy * n + gx] < FIRE_THRESHOLD:
                    continue
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx_, ny_ = gx + dx, gy + dy
                    if 0 <= nx_ < n and 0 <= ny_ < n:
                        if intensity[ny_ * n + nx_] < FIRE_THRESHOLD:
                            perimeter_cells.add((nx_, ny_))
        out = []
        for gx, gy in perimeter_cells:
            wx = (gx - n / 2.0 + 0.5) * cs
            wy = (gy - n / 2.0 + 0.5) * cs
            out.append((wx, wy))
        return out

    def _fire_centroid_world(self) -> Optional[tuple[float, float]]:
        if self._fire_intensity is None:
            return None
        n = self._grid_size
        cs = self._cell_size
        sx, sy, c = 0.0, 0.0, 0
        for gy in range(n):
            for gx in range(n):
                if self._fire_intensity[gy * n + gx] < FIRE_THRESHOLD:
                    continue
                wx = (gx - n / 2.0 + 0.5) * cs
                wy = (gy - n / 2.0 + 0.5) * cs
                sx += wx
                sy += wy
                c += 1
        if c == 0:
            return None
        return sx / c, sy / c

    def _start_next_perimeter_step(self) -> bool:
        """Pick the next point along the ordered fire ring and plan toward it."""
        if not self._grid_has_fire():
            return False

        ring = self._perimeter_world_points()
        if len(ring) < 2:
            return False

        cc = self._fire_centroid_world()
        if cc is None:
            return False
        cx, cy = cc
        ring.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

        rx, ry = self.current_position.x, self.current_position.y
        best_i = min(range(len(ring)), key=lambda i: math.hypot(ring[i][0] - rx, ring[i][1] - ry))
        stride = max(1, min(4, len(ring) // 12))
        gx, gy = ring[(best_i + stride) % len(ring)]

        prev = self._final_target
        self._final_target = (gx, gy)
        self._plan_path()
        planned = len(self._waypoints) > 0
        if not planned:
            self._final_target = prev
        return planned

    def _control_loop(self):
        self._control_tick += 1

        # Creep along the fire perimeter after a scout attack assignment
        if (
            self._perimeter_follow_active
            and not self._waypoints
            and self.target_position is None
            and self._final_target is None
            and self._fire_intensity is not None
        ):
            elapsed = time.monotonic() - self._last_perimeter_step_time
            if elapsed >= PERIMETER_STEP_INTERVAL_S:
                ok = self._start_next_perimeter_step()
                self._last_perimeter_step_time = time.monotonic()
                if not ok and not self._grid_has_fire():
                    self._perimeter_follow_active = False

        # Nothing to do
        if not self._waypoints and self.target_position is None:
            return

        # Periodic replan while navigating (fire spreads)
        if self._waypoints and self._final_target is not None:
            elapsed = time.monotonic() - self._last_plan_time
            if elapsed >= REPLAN_INTERVAL_S:
                self._plan_path()

            # If next waypoint is now on fire, replan immediately
            if self._waypoints and self._fire_intensity is not None:
                wx, wy = self._waypoints[0]
                gx, gy = _world_to_grid(
                    wx, wy, self._grid_size, self._cell_size)
                idx = gy * self._grid_size + gx
                if 0 <= idx < len(self._fire_intensity):
                    if self._fire_intensity[idx] >= FIRE_THRESHOLD:
                        self.get_logger().warn(
                            f'{self._tag} Next waypoint on fire — replanning')
                        self._plan_path()

        # If waypoints exist, drive toward the first one
        if self._waypoints:
            wx, wy = self._waypoints[0]
            dx = wx - self.current_position.x
            dy = wy - self.current_position.y
            distance = math.sqrt(dx ** 2 + dy ** 2)

            if distance < self.position_tolerance:
                self._waypoints.pop(0)
                if not self._waypoints:
                    # Reached final waypoint
                    cmd = Twist()
                    self.cmd_vel_pub.publish(cmd)
                    self.target_position = None
                    if self._pending_perimeter_after_scout:
                        self._perimeter_follow_active = True
                        self._pending_perimeter_after_scout = False
                        self._last_perimeter_step_time = (
                            time.monotonic() - PERIMETER_STEP_INTERVAL_S)
                    self._final_target = None
                    return
                wx, wy = self._waypoints[0]
                dx = wx - self.current_position.x
                dy = wy - self.current_position.y
                distance = math.sqrt(dx ** 2 + dy ** 2)

            # Check if the next step would enter fire
            if self._fire_intensity is not None:
                step_dist = min(self.linear_speed * 0.1, distance)
                if distance > 0.01:
                    step_x = self.current_position.x + (dx / distance) * step_dist
                    step_y = self.current_position.y + (dy / distance) * step_dist
                    sgx, sgy = _world_to_grid(
                        step_x, step_y, self._grid_size, self._cell_size)
                    idx = sgy * self._grid_size + sgx
                    if 0 <= idx < len(self._fire_intensity):
                        if self._fire_intensity[idx] >= FIRE_THRESHOLD:
                            self.get_logger().warn(
                                f'{self._tag} Would enter fire — stopping and replanning')
                            cmd = Twist()
                            self.cmd_vel_pub.publish(cmd)
                            self._plan_path()
                            return

            target_angle = math.atan2(dy, dx)
            angle_error = target_angle - self.current_yaw
            while angle_error > math.pi:
                angle_error -= 2 * math.pi
            while angle_error < -math.pi:
                angle_error += 2 * math.pi

            cmd = Twist()
            if abs(angle_error) > 0.2:
                cmd.angular.z = self.angular_speed if angle_error > 0 else -self.angular_speed
            else:
                cmd.linear.x = min(self.linear_speed, distance)
                cmd.angular.z = 0.5 * angle_error
            self.cmd_vel_pub.publish(cmd)
            return

        # Fallback: direct drive if no waypoints but target_position set
        if self.target_position is not None:
            dx = self.target_position.x - self.current_position.x
            dy = self.target_position.y - self.current_position.y
            distance = math.sqrt(dx ** 2 + dy ** 2)
            target_angle = math.atan2(dy, dx)

            angle_error = target_angle - self.current_yaw
            while angle_error > math.pi:
                angle_error -= 2 * math.pi
            while angle_error < -math.pi:
                angle_error += 2 * math.pi

            cmd = Twist()
            if distance < self.position_tolerance:
                self.target_position = None
            elif abs(angle_error) > 0.2:
                cmd.angular.z = self.angular_speed if angle_error > 0 else -self.angular_speed
            else:
                cmd.linear.x = min(self.linear_speed, distance)
                cmd.angular.z = 0.5 * angle_error
            self.cmd_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = NavigationController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
