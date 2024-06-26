import pygame
import random
import math
import torch
import matplotlib.pyplot as plt
import pandas as pd
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time as Timer
import os
import sys

import numpy as np

from DQN_agent import DQN_agent
from RL import Plane_rl

# Check if GPU is available
if torch.cuda.is_available():
    # If available, set the device to GPU
    device = torch.device("cuda")
    print("GPU available, using GPU for computations.")
else:
    # If not available, fall back to CPU
    device = torch.device("cpu")
    print("GPU not available, using CPU for computations.")

# Disable standard out to improve performance speed
performance = False
if performance: sys.stdout = open(os.devnull, 'w')

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)


# Define screen dimensions
WIDTH = 1500
HEIGHT = 800

class Airplane:
    def __init__(self, altitude, speed, accel=7.2, pitch_angle=30, drag_coeff=0.02, mass=1110, lift_coeff=0.5, color=WHITE):
        self.altitude = altitude
        self.speed = speed  # horizontal speed
        self.vertical_speed = 0
        self.throttle = 1.0
        self.accel = accel
        self.drag_c = drag_coeff
        self.mass = mass
        self.gravity = 9.81  # gravitational acceleration in m/s^2
        self.lift_coefficient = lift_coeff
        self.color = color
        self.elevator_angle = 0  # flap angle in degrees
        self.pitch_rate = 0  # degrees per second (rate of change)
        self.pitch_angle = pitch_angle  # in degrees (actual angle relative to world)

    def draw_airplane(self, screen, x=50):
        airplane_width, airplane_height = 30, 10
        airplane_surface = pygame.Surface((airplane_width, airplane_height), pygame.SRCALPHA)  # Ensure surface supports alpha for clean rotation
        airplane_surface.fill(self.color)
        rotated_surface = pygame.transform.rotate(airplane_surface, -self.pitch_angle)  # Negative for correct rotation direction
        rotated_rect = rotated_surface.get_rect(center=(x + airplane_width // 2, self.altitude + airplane_height // 2))

        screen.blit(rotated_surface, rotated_rect)


    def update(self, timediff):
        self.calculate_pitch(timediff)
        # self.calculate_speed(timediff)
        self.simple_speed(timediff)
        self.calculate_altitude(timediff)

    def calculate_pitch(self, timediff):
        # Simple pitch dynamics: changing the pitch rate based on elevator angle
        self.pitch_rate = 1 * self.elevator_angle  # coefficient determines responsiveness
        self.pitch_angle += self.pitch_rate * timediff
        if self.pitch_angle > 360 or self.pitch_angle < -360:
            self.pitch_angle = self.pitch_angle % 360  # normalize angle for simplicity

    def simple_speed(self, timediff):
        '''Uses simplified forces to be more understandable'''
        #engine power
        thrust = self.accel * self.throttle
        horz_thrust = thrust * math.cos(math.radians(self.pitch_angle))
        vert_thrust = - thrust * math.sin(math.radians(self.pitch_angle))

        #wing lift and drag
        vert_lift = 9.8 * math.cos(math.radians(self.pitch_angle)) #simple lift, when wing points flat to ground it is about as strong as gravity
       
        #drag to resist motion and non aerodynamic faces
        drag_c = -1.2
        horz_drag = drag_c * self.speed * (math.sin(math.radians(self.pitch_angle))+0.01) * timediff
        vert_drag = drag_c * self.vertical_speed * (math.cos(math.radians(self.pitch_angle))+0.01) * timediff

        #speeds
        self.speed = self.speed + ( horz_thrust - horz_drag) * timediff                                     #horizontal
        self.vertical_speed = self.vertical_speed + (vert_thrust - 9.8 + vert_lift - vert_drag) * timediff  #vertical


    def calculate_altitude(self, timediff):
        #NOTE: Pygame altitudes are switched
        self.altitude -= self.vertical_speed * timediff

    def controls(self, throttle, elevator_angle):
        #throttle limits
        if(throttle > 1):
            throttle = 1
        elif( throttle < 0):
            throttle = 0
        self.throttle = throttle

        #elevator flap limits
        if elevator_angle <= 0:
            self.elevator_angle = -20
        elif elevator_angle >= 1:
            self.elevator_angle = 20
        else:
            self.elevator_angle = (elevator_angle - 0.5) * 40

    def get_position(self):
        return self.speed, self.vertical_speed, self.altitude

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

def display_message(screen, text, color, x, y):
    font = pygame.font.Font(None, 36)  # Default font and size 36
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))

def plane_vectorize(plane, environment):
    '''
    Represents the airplane state as a vector of values
    '''
    #looks like (1, 0, 0 ... 120)
    return (
        plane.altitude,
        plane.throttle,
        plane.speed,
        plane.vertical_speed,
        plane.pitch_angle,
        plane.accel,
        plane.elevator_angle,
        plane.pitch_rate,
        environment[0],
        environment[1],
        environment[2]
    )

def main(csv_filename):
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Smooth Mountain Variation")

    clock = pygame.time.Clock()
    #1 second = 60 ticks
    #1 meter = 60 points
    if performance:
        tickspeed = 0  #unrestricted speed
    else:
        tickspeed #30 frames per second 
    experiences = []
    # Create an empty DataFrame
    plot_tracker = pd.DataFrame({
        "Altitude":[],
        "Speed":[],
        "Time":[]
    })
    start_time = Timer.time()
    # Initialize amplitude values for each segment
    amps = [65, 80, 20, 200, 130, 300, 40, 234, 100, 50, 78]

    game_over = False
    tick = 0    #tracks distances traveled in sets of 60
    mountain_points = generate_mountain_points(amps)
    count = 0

    ticks_per_meter = 60
    ticks_per_sec = 30
    
    # scoring
    distance_traveled = 0
    collision = False
    debug = False
    score = 0
    prev_speed = 65
    prev_angle = -1
    time = 0
    crash_message_decay = 0

    plane = Airplane(altitude=100, speed=prev_speed, pitch_angle=prev_angle)
    airplane_x = 50 #draw locations

    environment = (
        mountain_points[airplane_x],
        mountain_points[airplane_x+1],
        mountain_points[airplane_x+2]
        )
    airplane_vec = plane_vectorize(plane,environment)

    #build dimensions and assign random weights
    dims = [len(airplane_vec),78,54,20,2]
    # dims = [len(airplane_vec),46,25,10,2]
    plane_rl = Plane_rl(dims)
    ep = 0

    while not game_over:
        plane_height = plane.altitude - mountain_points[airplane_x]
        # details = "throttle: " + str(round(plane.throttle,2)) + ", \tspeed: " + str(round(plane.speed,2)) + " m/s, \taltitude: " + str(round(plane_height,2)) + "m"
        details = f"throttle: {round(plane.throttle, 2)}, speed: {round(plane.speed, 2)} m/s, altitude: {round(-plane_height, 2)}m"

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
                continue
        collision = False

        ## Airplane
        plane.update(1/ticks_per_sec)
        environment = (
            mountain_points[0],
            mountain_points[1],
            mountain_points[2]
        )
        airplane_vec_cur = plane_vectorize(plane, environment)

        # Create a new DataFrame with the current airplane data
        cur_frame = pd.DataFrame({
            "Altitude": [airplane_vec_cur[0]],
            "Speed": [airplane_vec_cur[2]],
            "Time": [Timer.time() - start_time]
        })

        # Concatenate the new DataFrame with the plot_tracker DataFrame
        plot_tracker = pd.concat([plot_tracker, cur_frame], ignore_index=True)

        output = plane_rl.get_action(ep, airplane_vec_cur)
        plane.controls(output[0],output[1])

        plane.update(1/ticks_per_sec)
        environment = (
            mountain_points[0],
            mountain_points[1],
            mountain_points[2]
        )
        airplane_vec_next = plane_vectorize(plane, environment)
        

        d_dist = plane.speed
        # d_dist = 1  #plane speed

        #prints every 60 ticks, meaning one tick every second on a tickspeed of 60
        #can be used to measure distance
        
        if debug and (tick !=0 and tick % 20 == 0):
            print("elevation: ", mountain_points[airplane_x] - plane.altitude)
            print("distance travelled: ", round(distance_traveled))
            print("airplane height: ", HEIGHT + plane.altitude)
            print("airplane speed: ", plane.speed)
            print("airplane pitch: ", plane.pitch_angle)
            print(details)

        print("Score: ", score)
        if  (tick !=0 and tick % 20 == 0):
            # print()
            state_b, target_q_values = plane_rl.generate_pattern_set(experiences)
            combined_data = list(zip(state_b, target_q_values))
            loss = plane_rl.train(combined_data)
            ep += 1

        #score metrics
        total_speed = math.sqrt(plane.speed ** 2 + plane.vertical_speed ** 2)
        # penalize change in values, 
        # priority to smooth flight, 
        # give points for staying within the optimal height
        d_speed = math.sqrt(prev_speed**2 + total_speed**2)
        d_angle = math.sqrt(prev_angle**2 + plane.pitch_angle**2)
        alt_metric = abs(plane.altitude - 100)/10
        offset = 40
        score = (- (d_angle) - (d_speed/plane.accel) - alt_metric + offset)/40
        # score_diff = oldscore - score

        distance_traveled = distance_traveled + d_dist / ticks_per_meter
        tick = (tick + 1) % 60

        if torch.is_tensor(d_dist):
            d_dist = d_dist.item()

        ## collision 
        mountain_points = mountain_points[round(d_dist):] + mountain_points[:round(d_dist)]  # Remove leftmost point
        if(plane.altitude >= mountain_points[airplane_x]):
            collision = True

        # Draw background
        screen.fill(BLACK)

        #draw airplane
        plane.draw_airplane(screen=screen,x=airplane_x)
        
        # Draw mountains
        pygame.draw.polygon(screen, GREEN, [(0, HEIGHT), *zip(range(WIDTH), mountain_points), (WIDTH, HEIGHT)])  

        crash_message_decay -= 1
        if collision: 

            if torch.is_tensor(score):
                score = score.item()
            if torch.is_tensor(distance_traveled):
                distance_traveled = distance_traveled.item()

            crash_message_decay = 60
            # display_message(screen, f"Score: {round(score + distance_traveled,ndigits=2)}", WHITE, WIDTH // 2, 20)

            # reset plane
            plane.altitude = 100
            plane.pitch_angle = -1
            plane.speed = 65
            plane.vertical_speed = 0
            score -= 100

        if crash_message_decay > 0 :
            display_message(screen, "Collision Detected!", RED, 50, HEIGHT // 2)
            # print("crash decay timer: ",crash_message_decay)

        experiences.append((airplane_vec_cur, output, score, airplane_vec_next, collision))      

        display_message(screen, details, WHITE, 10, 10)


        #break after 1000 ticks
        if not performance: pygame.display.flip()
        clock.tick(tickspeed) #frame speed
        count += 1
        if (count == 1000):
            break

    plot_tracker.to_csv(csv_filename, index=False)  # Set index=False to omit writing row indices to the file
    print("DataFrame successfully written to ", csv_filename)
    pygame.quit()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        main('test.csv')
    else:
        csv_filename = sys.argv[1]
        main(csv_filename)