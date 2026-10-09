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
    manual_mode: bool = False
    vehicle_index: int = 0
    throttle: float = 0.0
    brake: float = 0.0


@dataclass
class SensorStatus:
    name: str
    active: bool = True
    value: float = 0.0
    unit: str = "m"
    alert: str = "OK"


@dataclass(frozen=True)
class VehicleProfile:
    name: str
    color: tuple
    accent: tuple
    width: float
    height: float
    max_speed: float
    acceleration: float
    handling: float


VEHICLE_PROFILES = (
    VehicleProfile("Sport Sedan", (35, 125, 235), (110, 220, 255), 1.0, 1.0, 90.0, 28.0, 1.0),
    VehicleProfile("Urban SUV", (220, 65, 75), (255, 170, 80), 1.18, 1.15, 78.0, 23.0, 0.82),
    VehicleProfile("Electric Hatch", (35, 190, 130), (130, 255, 210), 0.9, 0.92, 82.0, 32.0, 1.12),
    VehicleProfile("Delivery Truck", (235, 150, 45), (255, 220, 120), 1.45, 1.35, 62.0, 15.0, 0.58),
    VehicleProfile("Rally Crossover", (150, 75, 225), (230, 160, 255), 1.08, 1.05, 86.0, 26.0, 0.95),
)

OBSTACLE_COLORS = {
    "cow": (125, 78, 42),
    "pothole": (35, 35, 42),
    "car": (60, 150, 220),
    "truck": (210, 130, 45),
    "bus": (190, 70, 160),
    "bike": (50, 205, 135),
}


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
        self.vehicle_index = 0
        self.mode_key_latched = False
        self.sensor_status = [
            SensorStatus("Camera", True, 12.0, "m", "Tracking"),
            SensorStatus("LiDAR", True, 25.0, "m", "Scanning"),
            SensorStatus("Radar", True, 18.0, "m", "Locked"),
            SensorStatus("Motion", True, 0.8, "g", "Stable"),
            SensorStatus("Ultrasonic", True, 1.7, "m", "Clear"),
        ]

    def reset(self):
        self.state = DriverState(vehicle_index=self.vehicle_index)
        self.obstacles.clear()
        self.spawn_timer = 0.0
        self.distance = 0.0

    @property
    def vehicle(self):
        return VEHICLE_PROFILES[self.state.vehicle_index]

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
        kind = random.choice(["cow", "pothole", "car", "truck", "bus", "bike"])
        size = random.uniform(0.8, 1.7)
        color = OBSTACLE_COLORS[kind]

        self.obstacles.append(
            Obstacle(
                kind=kind,
                z=ROAD_LENGTH,
                x=lane,
                size=size,
                speed=random.uniform(12.0, 28.0) if kind not in {"pothole", "bus"} else random.uniform(4.0, 18.0),
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

        if keys[pygame.K_m] and not self.mode_key_latched:
            self.state.manual_mode = not self.state.manual_mode
            self.message_timer = 1.6
            self.mode_key_latched = True
        elif not keys[pygame.K_m]:
            self.mode_key_latched = False

        for key, index in zip(
            (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5),
            range(len(VEHICLE_PROFILES)),
        ):
            if keys[key]:
                self.vehicle_index = index
                self.state.vehicle_index = index
                self.message_timer = 1.6
                pygame.time.delay(120)

        if self.state.paused:
            return

        profile = self.vehicle
        target_speed = self.state.speed
        if self.state.manual_mode:
            self.state.throttle = float(keys[pygame.K_w] or keys[pygame.K_UP])
            self.state.brake = float(keys[pygame.K_s] or keys[pygame.K_DOWN])
            target_speed += (profile.acceleration if self.state.throttle else -profile.acceleration * 0.35) * dt
            target_speed -= profile.acceleration * 1.35 * self.state.brake * dt
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                self.state.target_steer = -1.0
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                self.state.target_steer = 1.0
            else:
                self.state.target_steer = 0.0
        else:
            self.state.throttle = 1.0
            self.state.brake = 0.0
            target_speed += (profile.max_speed * 0.65 - target_speed) * min(1.0, 1.2 * dt)
            self.state.target_steer = -self.state.lateral / 2.8
            for obstacle in self.obstacles:
                if 3.0 < obstacle.z < 32.0 and abs(obstacle.x - self.state.lateral) < 1.5:
                    self.state.target_steer = -1.0 if obstacle.x >= self.state.lateral else 1.0
                    target_speed -= profile.acceleration * 1.5 * dt
                    break

        self.state.speed = max(0.0, min(profile.max_speed, target_speed))
        self.state.steer += (self.state.target_steer - self.state.steer) * min(1.0, 6.0 * profile.handling * dt)
        self.state.lateral += self.state.steer * 2.8 * profile.handling * dt * (self.state.speed / 18.0)
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
                collision_width = 0.75 * self.vehicle.width
                obstacle_width = 0.75 if obstacle.kind in {"truck", "bus"} else 0.55
                if abs(obstacle.x - self.state.lateral) < collision_width + obstacle_width:
                    self.state.health -= 22.0 if obstacle.kind == "pothole" else 35.0
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
        profile = self.vehicle
        width = 84 * profile.width
        height = 44 * profile.height
        car_rect = pygame.Rect(center_x - width / 2, car_y - height / 2, width, height)
        pygame.draw.rect(self.screen, profile.color, car_rect, border_radius=12)
        roof = pygame.Rect(center_x - width * 0.28, car_y - height * 0.43, width * 0.56, height * 0.42)
        pygame.draw.rect(self.screen, profile.accent, roof, border_radius=8)
        pygame.draw.rect(self.screen, (18, 30, 44), roof.inflate(-8, -5), border_radius=5)
        pygame.draw.rect(self.screen, (255, 245, 190), (center_x - width * 0.28, car_y - height * 0.12, width * 0.18, height * 0.18), border_radius=4)
        pygame.draw.rect(self.screen, (255, 245, 190), (center_x + width * 0.10, car_y - height * 0.12, width * 0.18, height * 0.18), border_radius=4)
        pygame.draw.rect(self.screen, (18, 20, 28), (center_x - width * 0.4, car_y + height * 0.25, width * 0.22, height * 0.3), border_radius=4)
        pygame.draw.rect(self.screen, (18, 20, 28), (center_x + width * 0.18, car_y + height * 0.25, width * 0.22, height * 0.3), border_radius=4)
        if self.state.manual_mode:
            pygame.draw.rect(self.screen, (80, 255, 210), car_rect.inflate(8, 8), 2, border_radius=14)

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
            elif obstacle.kind == "pothole":
                pothole = pygame.draw.circle(self.screen, obstacle.color, (int(sx), int(sy)), max(10, int(size * 0.75)))
                pygame.draw.circle(self.screen, (20, 20, 20), (int(sx), int(sy)), max(6, int(size * 0.4)), 2)
            else:
                vehicle_width = size * (52 if obstacle.kind in {"truck", "bus"} else 38) * scale
                vehicle_height = size * (34 if obstacle.kind in {"truck", "bus"} else 25) * scale
                body = pygame.Rect(sx - vehicle_width / 2, sy - vehicle_height, vehicle_width, vehicle_height)
                pygame.draw.rect(self.screen, obstacle.color, body, border_radius=max(3, int(6 * scale)))
                pygame.draw.rect(self.screen, (20, 35, 55), body.inflate(-vehicle_width * 0.25, -vehicle_height * 0.42), border_radius=4)
                pygame.draw.rect(self.screen, (255, 225, 120), (body.left + vehicle_width * 0.12, body.bottom - vehicle_height * 0.2, vehicle_width * 0.18, vehicle_height * 0.1))
                pygame.draw.rect(self.screen, (255, 225, 120), (body.right - vehicle_width * 0.3, body.bottom - vehicle_height * 0.2, vehicle_width * 0.18, vehicle_height * 0.1))

    def draw_hud(self):
        hud = pygame.Surface((330, 190), pygame.SRCALPHA)
        draw_rounded_rect(hud, (8, 15, 28, 220), (0, 0, 330, 190), 18)
        pygame.draw.rect(hud, self.vehicle.accent, (0, 0, 330, 3), border_radius=2)
        self.screen.blit(hud, (WINDOW_WIDTH - 355, 20))

        speed_text = self.font.render(f"{int(self.state.speed * 2.2)} km/h", True, (255, 255, 255))
        self.screen.blit(speed_text, (WINDOW_WIDTH - 320, 42))

        vehicle_text = self.small_font.render(f"{self.vehicle.name}  [{self.state.vehicle_index + 1}]", True, self.vehicle.accent)
        self.screen.blit(vehicle_text, (WINDOW_WIDTH - 320, 78))

        mode_text = self.small_font.render("MANUAL" if self.state.manual_mode else "AUTONOMOUS", True, (80, 255, 210) if self.state.manual_mode else (140, 190, 255))
        self.screen.blit(mode_text, (WINDOW_WIDTH - 320, 106))

        health_text = self.small_font.render(f"Health: {max(0, int(self.state.health))}%", True, (180, 240, 180))
        self.screen.blit(health_text, (WINDOW_WIDTH - 170, 106))

        score_text = self.small_font.render(f"Score: {int(self.state.score)}", True, (220, 220, 255))
        self.screen.blit(score_text, (WINDOW_WIDTH - 170, 134))

        control_text = self.tiny_font.render("M mode | 1-5 car | WASD drive | P pause | R reset", True, (200, 220, 255))
        self.screen.blit(control_text, (WINDOW_WIDTH - 340, 164))

        if self.state.crashed:
            crash = self.font.render("CRASHED", True, (255, 120, 120))
            self.screen.blit(crash, (WINDOW_WIDTH - 220, 140))
        elif self.state.paused:
            paused = self.font.render("PAUSED", True, (255, 224, 120))
            self.screen.blit(paused, (WINDOW_WIDTH - 210, 140))

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
