import pygame
import sys
import random

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

def get_height(dist):

   height = distances.get(dist)

   return height


out = net([23,34,54])

(throttle, elev_angle) = out