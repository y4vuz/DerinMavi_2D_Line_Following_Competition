import os
import sys
import argparse
import pygame
import numpy as np
import cv2
import importlib.util
import time
import math

# Configuration
WINDOW_WIDTH = 870
WINDOW_HEIGHT = 750
FPS = 60
TRACK_FILE = 'track.png'
CAR_FILE = 'racing_car.png'
SOLUTION_FILE = 'solution.py'

# Test Mode Constants
MAX_FRAMES = 18000  # Timeout (5 minutes at 60fps)
DT = 1.0 / 60.0    # Fixed time step

# Physics Constants
MAX_SPEED = 10.0
FRICTION = 0.98           # Rolling friction (velocity multiplier per frame)
ACCEL_RATE = 0.3          # Acceleration rate (reduced for realism)
BRAKE_RATE = 0.4          # Braking rate (how fast the car slows down)
TURN_RATE = 0.08          # Base turn rate (reduced for realism)

# Advanced Physics - Realism
LATERAL_FRICTION = 0.85   # How much lateral velocity is preserved (drift factor)
MASS = 1.0                # Car mass (affects momentum)
DRAG_COEFFICIENT = 0.02   # Air resistance
MIN_TURN_SPEED = 0.5      # Minimum speed needed to turn
TURN_SPEED_FACTOR = 0.15  # How much speed affects turning (higher = harder to turn at speed)
SLIP_ANGLE_THRESHOLD = 0.3  # When wheels start to slip
GRIP_LOSS_RATE = 0.7      # How much grip is lost when slipping

# Camera Sensor
CAMERA_OFFSET = 30
CAMERA_FOV_SIZE = (64, 64)

# Car Start Position
start_x, start_y = 50, 660
start_angle = -math.pi / 2


class Car:
    def __init__(self, x, y, angle, image=None):
        self.x = x
        self.y = y
        self.angle = angle
        
        # Velocity components (forward and lateral)
        self.velocity_forward = 0.0   # Speed along car's heading
        self.velocity_lateral = 0.0   # Speed perpendicular to heading (drift)
        
        # For compatibility
        self.speed = 0.0
        
        # Angular velocity
        self.angular_velocity = 0.0
        
        # Grip state
        self.is_slipping = False
        
        self.original_image = image
        if image:
            self.width = image.get_width()
            self.height = image.get_height()

    def update(self, steering, target_speed):
        # Clamp target speed
        target_speed = max(-MAX_SPEED * 0.3, min(MAX_SPEED, target_speed))
        
        # Calculate acceleration/braking
        speed_diff = target_speed - self.velocity_forward
        if speed_diff > 0:
            # Accelerating
            accel = min(ACCEL_RATE, speed_diff)
            self.velocity_forward += accel
        else:
            # Braking or coasting
            if target_speed < self.velocity_forward * 0.5:
                # Active braking
                decel = min(BRAKE_RATE, abs(speed_diff))
            else:
                # Coasting (engine braking)
                decel = min(ACCEL_RATE * 0.5, abs(speed_diff))
            self.velocity_forward -= decel
        
        # Apply rolling friction and air drag
        self.velocity_forward *= FRICTION
        self.velocity_forward -= self.velocity_forward * abs(self.velocity_forward) * DRAG_COEFFICIENT
        
        # Speed for compatibility
        self.speed = self.velocity_forward
        
        # Calculate effective turn rate based on speed
        # Turning is harder at high speeds, impossible when stopped
        if abs(self.velocity_forward) < MIN_TURN_SPEED:
            effective_turn_rate = TURN_RATE * (abs(self.velocity_forward) / MIN_TURN_SPEED)
        else:
            # At higher speeds, turning becomes harder
            speed_penalty = 1.0 / (1.0 + abs(self.velocity_forward) * TURN_SPEED_FACTOR)
            effective_turn_rate = TURN_RATE * speed_penalty
        
        # Calculate slip angle (difference between heading and velocity direction)
        requested_angular_change = steering * effective_turn_rate * abs(self.velocity_forward)
        
        # Check for wheel slip
        if abs(requested_angular_change) > SLIP_ANGLE_THRESHOLD:
            self.is_slipping = True
            # Reduce grip when slipping
            requested_angular_change *= GRIP_LOSS_RATE
            # Add lateral velocity (drift)
            drift_force = steering * abs(self.velocity_forward) * 0.1
            self.velocity_lateral += drift_force
        else:
            self.is_slipping = False
        
        # Apply angular velocity with smoothing
        target_angular_vel = requested_angular_change
        self.angular_velocity += (target_angular_vel - self.angular_velocity) * 0.3
        self.angle += self.angular_velocity
        
        # Apply lateral friction (reduces drift over time)
        self.velocity_lateral *= LATERAL_FRICTION
        
        # Calculate world velocity from forward and lateral components
        forward_x = self.velocity_forward * math.cos(self.angle)
        forward_y = self.velocity_forward * math.sin(self.angle)
        
        lateral_x = self.velocity_lateral * math.cos(self.angle + math.pi/2)
        lateral_y = self.velocity_lateral * math.sin(self.angle + math.pi/2)
        
        # Update position
        self.x += forward_x + lateral_x
        self.y += forward_y + lateral_y
        
        # Clamp speeds
        self.velocity_forward = max(-MAX_SPEED * 0.3, min(MAX_SPEED, self.velocity_forward))
        self.velocity_lateral = max(-MAX_SPEED * 0.5, min(MAX_SPEED * 0.5, self.velocity_lateral))

    def draw(self, surface):
        if self.original_image is None:
            return
        degrees = -math.degrees(self.angle)
        rotated_image = pygame.transform.rotate(self.original_image, degrees)
        rect = rotated_image.get_rect(center=(self.x, self.y))
        surface.blit(rotated_image, rect)
        
        # Draw slip indicator (optional debug visualization)
        if self.is_slipping:
            # Draw red circle when slipping
            pygame.draw.circle(surface, (255, 0, 0), (int(self.x), int(self.y) - 20), 5)

    def get_sensor_view(self, cv_track_img):
        cx = self.x + CAMERA_OFFSET * math.cos(self.angle)
        cy = self.y + CAMERA_OFFSET * math.sin(self.angle)

        patch_size = (CAMERA_FOV_SIZE[0], CAMERA_FOV_SIZE[1])
        M = cv2.getRotationMatrix2D((cx, cy), math.degrees(self.angle) + 90, 1.0)
        M[0, 2] += (patch_size[0] / 2) - cx
        M[1, 2] += (patch_size[1] / 2) - cy

        sensor_view = cv2.warpAffine(cv_track_img, M, patch_size)
        return sensor_view


def load_solution():
    try:
        spec = importlib.util.spec_from_file_location("solution", SOLUTION_FILE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"FAIL: Error loading solution.py: {e}")
        sys.exit(1)


def run_test_mode():
    """Headless test mode for CI/CD and leaderboard."""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    try:
        pygame.init()
        pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

        if not os.path.exists(TRACK_FILE):
            print("FAIL: track.png not found")
            sys.exit(1)

        cv_track = cv2.imread(TRACK_FILE)
        if cv_track is None:
            print("FAIL: Could not load track.png")
            sys.exit(1)

        solution = load_solution()
        car = Car(start_x, start_y, start_angle)

        frames = 0
        finished = False
        current_steering = 0.0

        while frames < MAX_FRAMES and not finished:
            frames += 1

            # Sensing (Color image for solution)
            sensor_view_color = car.get_sensor_view(cv_track)

            # Control
            try:
                target_speed, steering = solution.solution(sensor_view_color, car.speed, current_steering)
            except Exception as e:
                print(f"FAIL: Runtime error in solution: {e}")
                sys.exit(1)
            
            current_steering = steering

            # Update
            car.update(steering, target_speed)

            # Check Finish / Bounds / Obstacle
            cx, cy = int(car.x), int(car.y)
            if 0 <= cx < WINDOW_WIDTH and 0 <= cy < WINDOW_HEIGHT:
                pixel = cv_track[cy, cx]
                b, g, r = pixel

                # Yeşil bitiş noktası
                if g > 200 and r < 90 and b < 90:
                    finished = True
                    time_taken = frames * DT
                    print(f"FINAL_SCORE: {time_taken:.4f}")
                    sys.exit(0)
                
                # Kırmızı engele çarpma kontrolü
                if r > 200 and g < 50 and b < 50:
                    print("FAIL: Hit red obstacle")
                    sys.exit(1)
            else:
                print("FAIL: Out of bounds")
                sys.exit(1)

        if not finished:
            print("FAIL: Timeout")
            sys.exit(1)

    except Exception as e:
        print(f"FAIL: System error: {e}")
        sys.exit(1)
    finally:
        pygame.quit()


def run_dev_mode(debug=False):
    """Development mode with visual simulation."""
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Line Follower Challenge")
    clock = pygame.time.Clock()

    # Load Assets
    try:
        track_surface = pygame.image.load(TRACK_FILE).convert()
        car_image = pygame.image.load(CAR_FILE).convert_alpha()

        if car_image.get_width() > 50:
            car_image = pygame.transform.scale(
                car_image,
                (40, int(40 * car_image.get_height() / car_image.get_width()))
            )

        cv_track = cv2.imread(TRACK_FILE)

    except Exception as e:
        print(f"Failed to load assets: {e}")
        return

    # Initialize Car
    car = Car(start_x, start_y, start_angle, car_image)

    solution = load_solution()

    start_time = time.time()
    finish_time = None  # Bitiş zamanını kaydetmek için
    running = True
    finished = False
    current_steering = 0.0
    sensor_view_color = None

    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not finished:
            # 1. Get Sensor Data (Color image for solution)
            sensor_view_color = car.get_sensor_view(cv_track)

            # 2. Get Controls
            try:
                target_speed, steering = solution.solution(sensor_view_color, car.speed, current_steering)
            except Exception as e:
                print(f"Error in solution code: {e}")
                target_speed, steering = 0, 0
            
            current_steering = steering

            # 3. Update Physics
            car.update(steering, target_speed)

            # 4. Check Environment
            cx, cy = int(car.x), int(car.y)
            if 0 <= cx < WINDOW_WIDTH and 0 <= cy < WINDOW_HEIGHT:
                pixel = cv_track[cy, cx]
                b, g, r = pixel

                # Yeşil bitiş noktası
                if g > 200 and r < 90 and b < 90:
                    finish_time = time.time() - start_time  # Süreyi dondur
                    print(f"FINISHED! Time: {finish_time:.2f}s")
                    finished = True
                
                # Kırmızı engele çarpma kontrolü
                if r > 200 and g < 50 and b < 50:
                    print("HIT OBSTACLE! Game Over!")
                    running = False
            else:
                print("Out of bounds!")
                running = False

        # Rendering
        screen.blit(track_surface, (0, 0))
        car.draw(screen)

        # UI
        font = pygame.font.SysFont(None, 36)
        if finished and finish_time is not None:
            # Süre donduruldu, finish_time'ı göster
            msg = f"Finished! Time: {finish_time:.2f}s"
            color = (0, 255, 0)
        else:
            elapsed = time.time() - start_time
            msg = f"Time: {elapsed:.2f}s"
            color = (0, 0, 0)

        text = font.render(msg, True, color)
        screen.blit(text, (10, 10))

        # Debug: Show Camera View
        if debug and sensor_view_color is not None:
            # Convert BGR to RGB for pygame
            sensor_rgb = cv2.cvtColor(sensor_view_color, cv2.COLOR_BGR2RGB)
            # Transpose for pygame (height, width, channels) -> (width, height, channels)
            sensor_surf = pygame.surfarray.make_surface(sensor_rgb.swapaxes(0, 1))
            sensor_surf = pygame.transform.scale(sensor_surf, (128, 128))
            
            # Draw border
            border_rect = pygame.Rect(WINDOW_WIDTH - 140, 8, 132, 132)
            pygame.draw.rect(screen, (255, 255, 255), border_rect)
            pygame.draw.rect(screen, (0, 0, 0), border_rect, 2)
            
            screen.blit(sensor_surf, (WINDOW_WIDTH - 138, 10))
            
            # Label
            debug_font = pygame.font.SysFont(None, 20)
            debug_text = debug_font.render("Camera View", True, (0, 0, 0))
            screen.blit(debug_text, (WINDOW_WIDTH - 138, 145))

        pygame.display.flip()

    pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description="Line Follower Challenge - Robot Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              # Development mode (visual)
  python main.py --debug      # Development mode with camera view
  python main.py --test       # Test mode (headless, for CI/CD)
        """
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in headless test mode (for CI/CD and leaderboard)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Show camera sensor view in development mode'
    )
    
    args = parser.parse_args()
    
    if args.test:
        if args.debug:
            print("Warning: --debug is ignored in test mode")
        run_test_mode()
    else:
        run_dev_mode(debug=args.debug)


if __name__ == "__main__":
    main()
