import pygame
import sys
import random
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

# Game settings
ground_heights = []
number_of_points = 100  # You can adjust this for more or less detail in the ground

class Airplane:
    def __init__(self, altitude, speed, accel=7.2, drag_coeff=0.02, mass=1110, lift_coeff = 0.5):
        self.altitude = altitude
        self.speed = speed  # This will now represent the horizontal speed
        self.vertical_speed = 0
        self.pitch_angle = 0  # In degrees
        self.throttle = 1
        self.accel = accel
        self.drag_c = drag_coeff
        self.mass = mass
        self.gravity = 9.81  # m/s^2, downward force
        self.lift_coefficient = lift_coeff

    def update(self, timediff, throttle, elevator_angle):
        self.controls(throttle, elevator_angle)
        self.calculate_speed(timediff)
        self.calculate_altitude(timediff)

    def controls(self, throttle, elevator_angle):
        self.throttle = throttle
        self.pitch_angle = elevator_angle

    def calculate_speed(self, timediff):
        # Calculate the engine force
        engine_force = self.accel * self.throttle

        # Calculate the drag
        drag = self.drag_c * (self.speed ** 2)

        # Net horizontal force
        horizontal_force = engine_force - drag
        horizontal_acceleration = horizontal_force / self.mass

        # Update horizontal speed
        self.speed += horizontal_acceleration * timediff

        # Lift calculation
        lift = self.lift_coefficient * (self.speed ** 2) * math.cos(math.radians(self.pitch_angle))

        # Vertical force calculation
        net_vertical_force = lift - self.mass * self.gravity
        vertical_acceleration = net_vertical_force / self.mass

        # Update vertical speed
        self.vertical_speed += vertical_acceleration * timediff

    def calculate_altitude(self, timediff):
        # Update altitude
        self.altitude += self.vertical_speed * timediff

    def get_position(self):
        return self.speed, self.vertical_speed, self.altitude 

def generate_ground():
    """Generates varying ground heights."""
    for i in range(number_of_points):
        # Example: Simple sine wave; replace with any function you like
        height = SCREEN_HEIGHT / 2 + (random.randint(-50, 50))  # Random for simplicity
        ground_heights.append(height)

def draw_ground():
    """Draws the ground based on heights."""
    for i, height in enumerate(ground_heights):
        pygame.draw.line(screen, GREEN, 
                         (i * (SCREEN_WIDTH / (number_of_points - 1)), SCREEN_HEIGHT), 
                         (i * (SCREEN_WIDTH / (number_of_points - 1)), height), 5)

def game_loop():
    clock = pygame.time.Clock()
    generate_ground()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill(WHITE)
        draw_ground()
        
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    game_loop()