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
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60
TRACK_FILE = 'track.png'
CAR_FILE = 'racing_car.png'
SOLUTION_FILE = 'solution.py'

# Test Mode Constants
MAX_FRAMES = 18000  # Timeout (5 minutes at 60fps)
DT = 1.0 / 60.0    # Fixed time step

# Physics Constants
MAX_SPEED = 10.0
FRICTION = 0.5
ACCEL_RATE = 0.5
TURN_RATE = 0.1

# Camera Sensor
CAMERA_OFFSET = 30
CAMERA_FOV_SIZE = (64, 64)

# Car Start Position
start_x, start_y = 60, 550
start_angle = -math.pi / 2


class Car:
    def __init__(self, x, y, angle, image=None):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 0.0
        self.original_image = image
        if image:
            self.width = image.get_width()
            self.height = image.get_height()

    def update(self, steering, target_speed):
        if self.speed < target_speed:
            self.speed += ACCEL_RATE
        elif self.speed > target_speed:
            self.speed -= ACCEL_RATE

        self.x += self.speed * math.cos(self.angle)
        self.y += self.speed * math.sin(self.angle)
        self.angle += steering * self.speed * TURN_RATE

    def draw(self, surface):
        if self.original_image is None:
            return
        degrees = -math.degrees(self.angle)
        rotated_image = pygame.transform.rotate(self.original_image, degrees)
        rect = rotated_image.get_rect(center=(self.x, self.y))
        surface.blit(rotated_image, rect)

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
        car = Car(100, 300, -math.pi / 2)

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

            # Check Finish / Bounds
            cx, cy = int(car.x), int(car.y)
            if 0 <= cx < WINDOW_WIDTH and 0 <= cy < WINDOW_HEIGHT:
                pixel = cv_track[cy, cx]
                b, g, r = pixel

                if g > 200 and r < 50 and b < 50:
                    finished = True
                    time_taken = frames * DT
                    print(f"FINAL_SCORE: {time_taken:.4f}")
                    sys.exit(0)
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

                if g > 200 and r < 50 and b < 50:
                    print(f"FINISHED! Time: {time.time() - start_time:.2f}s")
                    finished = True
            else:
                print("Out of bounds!")
                running = False

        # Rendering
        screen.blit(track_surface, (0, 0))
        car.draw(screen)

        # UI
        font = pygame.font.SysFont(None, 36)
        elapsed = time.time() - start_time
        if finished:
            msg = f"Finished! Time: {elapsed:.2f}s"
            color = (0, 255, 0)
        else:
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
