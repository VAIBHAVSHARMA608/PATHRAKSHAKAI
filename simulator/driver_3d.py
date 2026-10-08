import math
import random
import time
from dataclasses import dataclass, field

import pygame


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
HORIZON_Y = 180
ROAD_HALF_WIDTH = 420
ROAD_LENGTH = 130


@dataclass
class Obstacle:
    kind: str
    z: float
    x: float
    size: float = 1.0
    speed: float = 0.0
    color: tuple = (255, 255, 255)
    wobble: float = 0.0
    active: bool = True


@dataclass
class DriverState:
    lateral: float = 0.0
    speed: float = 28.0
    steer: float = 0.0
    target_steer: float = 0.0
    score: float = 0.0
    health: float = 100.0
    crashed: bool = False
    paused: bool = False


@dataclass
class SensorStatus:
    name: str
    active: bool = True
    value: float = 0.0
    unit: str = "m"
    alert: str = "OK"


def draw_rounded_rect(surface, color, rect, radius):
    x, y, w, h = rect
    if w < 2 * radius:
        radius = w // 2
    if h < 2 * radius:
        radius = h // 2

    pygame.draw.rect(surface, color, (x + radius, y, w - 2 * radius, h))
    pygame.draw.rect(surface, color, (x, y + radius, w, h - 2 * radius))
    pygame.draw.circle(surface, color, (x + radius, y + radius), radius)
    pygame.draw.circle(surface, color, (x + w - radius, y + radius), radius)
    pygame.draw.circle(surface, color, (x + radius, y + h - radius), radius)
    pygame.draw.circle(surface, color, (x + w - radius, y + h - radius), radius)


class Driver3DScene:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("PathRakshak 3D Driver View")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Segoe UI", 24)
        self.small_font = pygame.font.SysFont("Segoe UI", 18)
        self.tiny_font = pygame.font.SysFont("Segoe UI", 14)
        self.state = DriverState()
        self.obstacles: list[Obstacle] = []
        self.spawn_timer = 0.0
        self.distance = 0.0
        self.running = True
        self.last_time = time.time()
        self.sky_top = (9, 16, 26)
        self.sky_bottom = (38, 61, 90)
        self.message_timer = 0.0
        self.sensor_status = [
            SensorStatus("Camera", True, 12.0, "m", "Tracking"),
            SensorStatus("LiDAR", True, 25.0, "m", "Scanning"),
            SensorStatus("Radar", True, 18.0, "m", "Locked"),
            SensorStatus("Motion", True, 0.8, "g", "Stable"),
            SensorStatus("Ultrasonic", True, 1.7, "m", "Clear"),
        ]

    def reset(self):
        self.state = DriverState()
        self.obstacles.clear()
        self.spawn_timer = 0.0
        self.distance = 0.0

    def project(self, world_x, world_z):
        """Perspective projection for pseudo-3D world coordinates."""
        depth = max(0.2, min(1.0, world_z / ROAD_LENGTH))
        scale = 1.0 / (0.25 + depth)
        sx = WINDOW_WIDTH * 0.5 + (world_x * 280.0) * scale
        sy = HORIZON_Y + (1.0 - depth) * (WINDOW_HEIGHT - HORIZON_Y)
        return sx, sy, scale

    def spawn_obstacle(self):
        lane_choices = [-2.4, -1.2, 0.0, 1.2, 2.4]
        lane = random.choice(lane_choices)
        kind = random.choice(["cow", "pothole"])
        size = random.uniform(0.8, 1.7)

        if kind == "cow":
            color = (92, 61, 35)
        else:
            color = (38, 38, 40)

        self.obstacles.append(
            Obstacle(
                kind=kind,
                z=ROAD_LENGTH,
                x=lane,
                size=size,
                speed=random.uniform(12.0, 28.0),
                color=color,
                wobble=random.uniform(0.0, math.tau),
            )
        )

    def handle_input(self, dt):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_p]:
            self.state.paused = not self.state.paused
            self.message_timer = 1.6
            pygame.time.delay(150)

        if keys[pygame.K_r]:
            self.reset()
            pygame.time.delay(150)

        if self.state.paused:
            return

        target_speed = self.state.speed
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            target_speed += 26.0 * dt
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            target_speed -= 35.0 * dt
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.state.target_steer = -1.0
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.state.target_steer = 1.0
        else:
            self.state.target_steer = 0.0

        self.state.speed = max(0.0, min(90.0, target_speed))
        self.state.steer += (self.state.target_steer - self.state.steer) * min(1.0, 6.0 * dt)
        self.state.lateral += self.state.steer * 2.8 * dt * (self.state.speed / 18.0)
        self.state.lateral = max(-3.5, min(3.5, self.state.lateral))

    def update(self, dt):
        if self.state.crashed:
            return

        self.handle_input(dt)
        if self.state.paused:
            return

        self.distance += self.state.speed * dt

        self.spawn_timer -= dt
        if self.spawn_timer <= 0:
            self.spawn_obstacle()
            self.spawn_timer = max(0.5, 1.5 - self.state.speed / 120.0 + random.uniform(0.2, 0.8))

        for obstacle in self.obstacles[:]:
            obstacle.z -= (self.state.speed * dt) * (0.75 + obstacle.speed / 60.0)
            obstacle.wobble += dt * 2.5

            if obstacle.kind == "cow":
                obstacle.x += math.sin(obstacle.wobble) * 0.08
            if obstacle.kind == "pothole":
                obstacle.x += math.sin(obstacle.wobble * 1.5) * 0.04

            if obstacle.z < 2.0:
                if abs(obstacle.x - self.state.lateral) < 0.9:
                    self.state.health -= 35.0 if obstacle.kind == "pothole" else 50.0
                    self.state.score = max(0.0, self.state.score - 15.0)
                    obstacle.active = False
                    if obstacle.kind == "pothole":
                        self.state.speed *= 0.7
                if obstacle.z < -10:
                    self.obstacles.remove(obstacle)
                    self.state.score += 5.0

        if self.state.health <= 0:
            self.state.crashed = True

        # Update sensor status based on proximity and motion.
        nearest = 999.0
        for obstacle in self.obstacles:
            if obstacle.active:
                nearest = min(nearest, abs(obstacle.z))

        camera_dist = max(0.0, 15.0 - self.state.speed / 5.0)
        lidar_dist = max(0.0, 35.0 - nearest)
        radar_dist = max(0.0, 25.0 - nearest * 0.5)
        motion = abs(self.state.steer) * 0.7 + abs(self.state.speed) / 80.0

        self.sensor_status[0].value = camera_dist
        self.sensor_status[0].alert = "Tracking" if camera_dist < 12 else "Clear"
        self.sensor_status[1].value = lidar_dist
        self.sensor_status[1].alert = "Obstacle" if lidar_dist < 8 else "Scanning"
        self.sensor_status[2].value = radar_dist
        self.sensor_status[2].alert = "Moving" if radar_dist < 10 else "Locked"
        self.sensor_status[3].value = motion
        self.sensor_status[3].alert = "Shift" if motion > 0.8 else "Stable"
        self.sensor_status[4].value = min(5.0, max(0.0, nearest / 10.0))
        self.sensor_status[4].alert = "Close" if self.sensor_status[4].value < 1.2 else "Clear"

    def draw_road(self):
        screen = self.screen
        screen.fill(self.sky_top)

        for i in range(20):
            y1 = HORIZON_Y + i * 20
            y2 = HORIZON_Y + (i + 1) * 20
            alpha = i / 20.0
            color = (
                int(self.sky_top[0] + (self.sky_bottom[0] - self.sky_top[0]) * alpha),
                int(self.sky_top[1] + (self.sky_bottom[1] - self.sky_top[1]) * alpha),
                int(self.sky_top[2] + (self.sky_bottom[2] - self.sky_top[2]) * alpha),
            )
            pygame.draw.rect(screen, color, (0, y1, WINDOW_WIDTH, 20))

        road_top = 200
        road_bottom = WINDOW_HEIGHT
        road_left_top = WINDOW_WIDTH * 0.5 - ROAD_HALF_WIDTH * 0.3
        road_right_top = WINDOW_WIDTH * 0.5 + ROAD_HALF_WIDTH * 0.3
        road_left_bottom = WINDOW_WIDTH * 0.5 - ROAD_HALF_WIDTH
        road_right_bottom = WINDOW_WIDTH * 0.5 + ROAD_HALF_WIDTH

        pygame.draw.polygon(
            screen,
            (32, 36, 38),
            [(road_left_top, HORIZON_Y), (road_right_top, HORIZON_Y), (road_right_bottom, road_bottom), (road_left_bottom, road_bottom)],
        )

        shoulder_color = (60, 65, 62)
        pygame.draw.polygon(
            screen,
            shoulder_color,
            [(road_left_top - 20, HORIZON_Y), (road_right_top + 20, HORIZON_Y), (road_right_bottom + 18, road_bottom), (road_left_bottom - 18, road_bottom)],
        )

        for i in range(1, 11):
            t = i / 11.0
            x1 = self.lerp(road_left_top, road_left_bottom, t)
            x2 = self.lerp(road_right_top, road_right_bottom, t)
            y1 = self.lerp(HORIZON_Y, road_bottom, t)
            y2 = self.lerp(HORIZON_Y, road_bottom, t)
            pygame.draw.line(screen, (200, 200, 200), (x1, y1), (x2, y2), 2)

        lane_markers = 18
        lane_offset = (self.distance * 0.9) % 8.0
        for i in range(lane_markers):
            frac = i / lane_markers
            y = HORIZON_Y + frac * (WINDOW_HEIGHT - HORIZON_Y)
            dash = max(6, int(32 * (1.0 - frac)))
            left = self.lerp(road_left_top, road_left_bottom, frac)
            right = self.lerp(road_right_top, road_right_bottom, frac)
            px = left + (right - left) * 0.5
            pygame.draw.rect(screen, (230, 230, 230), (px - 2, y + lane_offset * 7, 4, dash), border_radius=2)

    def lerp(self, a, b, t):
        return a + (b - a) * t

    def draw_car(self):
        center_x = WINDOW_WIDTH * 0.5 + self.state.lateral * 140.0
        car_y = WINDOW_HEIGHT - 110
        car_rect = pygame.Rect(center_x - 42, car_y - 22, 84, 44)
        pygame.draw.rect(self.screen, (20, 130, 230), car_rect, border_radius=12)
        pygame.draw.rect(self.screen, (255, 255, 255), (center_x - 18, car_y - 8, 14, 16), border_radius=4)
        pygame.draw.rect(self.screen, (255, 255, 255), (center_x + 4, car_y - 8, 14, 16), border_radius=4)
        pygame.draw.rect(self.screen, (22, 22, 30), (center_x - 32, car_y + 8, 20, 16), border_radius=4)
        pygame.draw.rect(self.screen, (22, 22, 30), (center_x + 12, car_y + 8, 20, 16), border_radius=4)

    def draw_obstacles(self):
        for obstacle in self.obstacles:
            if not obstacle.active:
                continue
            sx, sy, scale = self.project(obstacle.x - self.state.lateral, obstacle.z)
            size = max(12, obstacle.size * 42 * scale)
            if obstacle.kind == "cow":
                body = pygame.Rect(sx - size * 0.9, sy - size * 0.45, size * 1.8, size * 0.9)
                pygame.draw.ellipse(self.screen, obstacle.color, body)
                pygame.draw.ellipse(self.screen, (255, 255, 255), body.inflate(-12, -12), 2)
                pygame.draw.circle(self.screen, (22, 22, 22), (int(sx - size * 0.3), int(sy - 4)), 4)
                pygame.draw.circle(self.screen, (22, 22, 22), (int(sx + size * 0.2), int(sy - 4)), 4)
            else:
                pothole = pygame.draw.circle(self.screen, obstacle.color, (int(sx), int(sy)), max(10, int(size * 0.75)))
                pygame.draw.circle(self.screen, (20, 20, 20), (int(sx), int(sy)), max(6, int(size * 0.4)), 2)

    def draw_hud(self):
        hud = pygame.Surface((260, 148), pygame.SRCALPHA)
        draw_rounded_rect(hud, (15, 18, 24, 180), (0, 0, 260, 148), 16)
        self.screen.blit(hud, (WINDOW_WIDTH - 285, 20))

        speed_text = self.font.render(f"{int(self.state.speed * 2.2)} km/h", True, (255, 255, 255))
        self.screen.blit(speed_text, (WINDOW_WIDTH - 240, 44))

        health_text = self.small_font.render(f"Health: {max(0, int(self.state.health))}%", True, (180, 240, 180))
        self.screen.blit(health_text, (WINDOW_WIDTH - 245, 80))

        score_text = self.small_font.render(f"Score: {int(self.state.score)}", True, (220, 220, 255))
        self.screen.blit(score_text, (WINDOW_WIDTH - 245, 102))

        control_text = self.tiny_font.render("W/S steer | A/D turn | P pause | R reset", True, (200, 220, 255))
        self.screen.blit(control_text, (WINDOW_WIDTH - 270, 126))

        if self.state.crashed:
            crash = self.font.render("CRASHED", True, (255, 120, 120))
            self.screen.blit(crash, (WINDOW_WIDTH - 210, 124))
        elif self.state.paused:
            paused = self.font.render("PAUSED", True, (255, 224, 120))
            self.screen.blit(paused, (WINDOW_WIDTH - 200, 124))

    def draw_sensor_panel(self):
        panel_x = 20
        panel_y = 20
        panel_w = 290
        panel_h = 260
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        draw_rounded_rect(panel, (12, 16, 24, 200), (0, 0, panel_w, panel_h), 16)
        self.screen.blit(panel, (panel_x, panel_y))

        title = self.small_font.render("Perception Stack", True, (255, 255, 255))
        self.screen.blit(title, (panel_x + 18, panel_y + 14))

        row_h = 38
        for idx, sensor in enumerate(self.sensor_status):
            py = panel_y + 52 + idx * row_h
            dot_color = (40, 220, 120) if sensor.active and sensor.alert in {"Tracking", "Scanning", "Locked", "Stable", "Clear"} else (255, 120, 120)
            pygame.draw.circle(self.screen, dot_color, (panel_x + 30, py + 12), 6)

            name_text = self.tiny_font.render(sensor.name, True, (230, 230, 230))
            self.screen.blit(name_text, (panel_x + 46, py + 4))

            value_text = self.tiny_font.render(f"{sensor.value:.1f}{sensor.unit}", True, (255, 255, 255))
            self.screen.blit(value_text, (panel_x + 140, py + 4))

            alert_text = self.tiny_font.render(sensor.alert, True, (160, 220, 255))
            self.screen.blit(alert_text, (panel_x + 200, py + 4))

            pygame.draw.line(self.screen, (70, 80, 90), (panel_x + 18, py + 26), (panel_x + panel_w - 18, py + 26), 1)

    def draw_scene(self):
        self.draw_road()
        self.draw_obstacles()
        self.draw_car()
        self.draw_sensor_panel()
        self.draw_hud()

        if self.state.paused:
            overlay = pygame.Surface((220, 60), pygame.SRCALPHA)
            draw_rounded_rect(overlay, (20, 24, 32, 200), (0, 0, 220, 60), 12)
            self.screen.blit(overlay, (WINDOW_WIDTH // 2 - 110, WINDOW_HEIGHT // 2 - 30))
            label = self.font.render("Paused", True, (255, 255, 255))
            self.screen.blit(label, (WINDOW_WIDTH // 2 - 52, WINDOW_HEIGHT // 2 - 18))

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.running = False

            self.update(dt)
            self.draw_scene()
            pygame.display.flip()

        pygame.quit()


def run_driver_demo():
    game = Driver3DScene()
    game.run()


if __name__ == "__main__":
    run_driver_demo()
