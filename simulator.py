import pygame
import random
import math

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

# Define screen dimensions
WIDTH = 1000
HEIGHT = 800

# Function to generate mountain points with smooth amplitude variation
def generate_mountain_points(amps):
    points = []
    num_segments = len(amps)
    segment_width = WIDTH // num_segments
    a = 3
    for x in range(WIDTH):
        segment_index = min(x // segment_width, num_segments - 1)
        x0 = segment_index * segment_width
        x1 = min((segment_index + 1) * segment_width, WIDTH - 1)
        t = (x - x0) / (x1 - x0)  # Interpolation parameter between 0 and 1
        y0 = amps[segment_index] * (math.sin(a * math.pi * x0 / WIDTH)) + HEIGHT // 2
        y1 = amps[(segment_index + 1) % num_segments] * (math.sin(a * math.pi * x1 / WIDTH)) + HEIGHT // 2
        y = (1 - t) * y0 + t * y1  # Linear interpolation
        points.append(y)

    return points

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Smooth Mountain Variation")

    clock = pygame.time.Clock()

    # Initialize amplitude values for each segment
    amps = [65, 80, 20, 200, 130, 300]

    game_over = False
    mountain_points = generate_mountain_points(amps)

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True

        # Generate mountain points with smooth variation
        # points = generate_mountain_points(amps)
        mountain_points = mountain_points[1:] + [mountain_points[0]]  # Remove leftmost point

        # Draw background
        screen.fill(BLACK)

        # Draw mountains
        pygame.draw.polygon(screen, GREEN, [(0, HEIGHT), *zip(range(WIDTH), mountain_points), (WIDTH, HEIGHT)])

        pygame.display.flip()
        clock.tick(300) #frame speed

    pygame.quit()

if __name__ == "__main__":
    main()
