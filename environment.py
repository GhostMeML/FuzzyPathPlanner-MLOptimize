import numpy as np
import random
import config


class SimulationEnvironment:
    def __init__(self):
        self.robot_pos, self.target_pos, self.obstacles = self._generate_env()
        self.robot_vel = np.array([0.0, 0.0])
        self.robot_angle = 0.0

    def _generate_env(self):
        robot_pos = np.array([random.uniform(1, 5), random.uniform(1, 5)])
        target_pos = np.array([random.uniform(15, 19), random.uniform(15, 19)])
        obstacles = []

        def valid_pos(p):
            return (np.linalg.norm(p - robot_pos) > 4 and
                    np.linalg.norm(p - target_pos) > 4 and
                    2 < p[0] < config.WIDTH - 2 and 2 < p[1] < config.HEIGHT - 2)

        while len(obstacles) < config.NUM_OBSTACLES:
            p = np.array([random.uniform(3, 17), random.uniform(3, 17)])
            if valid_pos(p) and all(np.linalg.norm(p - o['pos']) > 3 for o in obstacles):
                vel = np.random.uniform(-0.15, 0.15, 2)
                obstacles.append({'pos': p, 'vel': vel})

        return robot_pos, target_pos, obstacles

    def get_sensor_distances(self):
        dirs = [(self.robot_angle + np.pi / 2, 'left'),
                (self.robot_angle, 'front'),
                (self.robot_angle - np.pi / 2, 'right')]
        dists = {}

        for ang, name in dirs:
            min_dist = config.SENSOR_RANGE
            ray = np.array([np.cos(ang), np.sin(ang)])

            for obs in self.obstacles:
                to_obs = obs['pos'] - self.robot_pos
                d = np.linalg.norm(to_obs)
                if d >= min_dist: continue

                proj = np.dot(to_obs, ray)
                if proj > 0:
                    perp = np.linalg.norm(to_obs - proj * ray)
                    if perp < config.OBSTACLE_RADIUS + 0.3:
                        min_dist = min(min_dist, max(0, proj - config.OBSTACLE_RADIUS))

            # Проверка границ поля
            if abs(np.cos(ang)) > 0.7:
                wall_dist = self.robot_pos[0] if np.cos(ang) < 0 else config.WIDTH - self.robot_pos[0]
                min_dist = min(min_dist, wall_dist)
            if abs(np.sin(ang)) > 0.7:
                wall_dist = self.robot_pos[1] if np.sin(ang) < 0 else config.HEIGHT - self.robot_pos[1]
                min_dist = min(min_dist, wall_dist)

            dists[name] = min_dist

        return dists['left'], dists['front'], dists['right']

    def get_target_angle(self):
        to_target = self.target_pos - self.robot_pos
        target_ang = np.arctan2(to_target[1], to_target[0])
        rel = target_ang - self.robot_angle
        rel = (rel + np.pi) % (2 * np.pi) - np.pi
        return np.degrees(rel)

    def step(self, turn_angle, speed_val):
        # 1. Обновление препятствий
        for obs in self.obstacles:
            obs['pos'] += obs['vel'] * config.DT * 5
            if obs['pos'][0] <= config.OBSTACLE_RADIUS or obs['pos'][0] >= config.WIDTH - config.OBSTACLE_RADIUS:
                obs['vel'][0] *= -1
            if obs['pos'][1] <= config.OBSTACLE_RADIUS or obs['pos'][1] >= config.HEIGHT - config.OBSTACLE_RADIUS:
                obs['vel'][1] *= -1
            obs['pos'] = np.clip(obs['pos'], config.OBSTACLE_RADIUS,
                                 [config.WIDTH - config.OBSTACLE_RADIUS, config.HEIGHT - config.OBSTACLE_RADIUS])

        # 2. Движение робота
        target_angle = self.robot_angle + np.radians(turn_angle)
        angle_diff = (target_angle - self.robot_angle + np.pi) % (2 * np.pi) - np.pi
        self.robot_angle += angle_diff * 0.3 * config.DT * 10

        speed_target = speed_val * config.MAX_SPEED
        self.robot_vel *= 0.85
        self.robot_vel += np.array([np.cos(self.robot_angle), np.sin(self.robot_angle)]) * speed_target * config.DT * 5

        new_pos = self.robot_pos + self.robot_vel * config.DT

        # Проверка границ для робота
        if (new_pos[0] < config.ROBOT_RADIUS or new_pos[0] > config.WIDTH - config.ROBOT_RADIUS or
                new_pos[1] < config.ROBOT_RADIUS or new_pos[1] > config.HEIGHT - config.ROBOT_RADIUS):
            self.robot_vel *= -0.5
            self.robot_angle += np.pi / 2
        else:
            self.robot_pos = new_pos

        self._resolve_collisions()

    def _resolve_collisions(self):
        for obs in self.obstacles:
            diff = self.robot_pos - obs['pos']
            dist = np.linalg.norm(diff)
            min_dist = config.ROBOT_RADIUS + config.OBSTACLE_RADIUS

            if dist < min_dist:
                overlap = min_dist - dist
                push_dir = diff / dist

                self.robot_pos += push_dir * overlap * 2.5
                self.robot_vel = -self.robot_vel * 0.8
                self.robot_vel += push_dir * 3.0

                angle_to_obs = np.arctan2(diff[1], diff[0])
                self.robot_angle = angle_to_obs + np.pi / 2 + np.random.uniform(-0.5, 0.5)

    def is_goal_reached(self):
        return np.linalg.norm(self.robot_pos - self.target_pos) < config.TARGET_RADIUS + config.ROBOT_RADIUS