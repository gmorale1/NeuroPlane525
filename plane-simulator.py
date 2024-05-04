import pygame
import random
import math
import torch
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from DQN_agent import DQN_agent
from RL import Plane_rl


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
        self.pitch_angle = pitch_angle  # in degrees
        self.throttle = 1
        self.accel = accel
        self.drag_c = drag_coeff
        self.mass = mass
        self.gravity = 9.81  # gravitational acceleration in m/s^2
        self.lift_coefficient = lift_coeff
        self.color = color
        self.elevator_angle = 0  # in degrees
        self.pitch_rate = 0  # degrees per second

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
        self.pitch_rate = 2 * self.elevator_angle  # coefficient determines responsiveness
        self.pitch_angle += self.pitch_rate * timediff
        if self.pitch_angle > 360 or self.pitch_angle < -360:
            self.pitch_angle = self.pitch_angle % 360  # normalize angle for simplicity

    def calculate_speed(self, timediff):
        # Calculate total engine force
        engine_force = self.accel * self.throttle

        # Convert pitch angle to radians
        pitch_radians = math.radians(self.pitch_angle)

        # Decompose the total engine force into horizontal and vertical components
        horizontal_thrust = engine_force * math.cos(pitch_radians)
        vertical_thrust = engine_force * math.sin(pitch_radians)

        # Calculate drag based on total speed (hypotenuse of horizontal and vertical speeds)
        total_speed = math.sqrt(self.speed ** 2 + self.vertical_speed ** 2)
        drag = self.drag_c * (total_speed ** 2)

        # Calculate horizontal forces and update horizontal speed
        horizontal_net_force = horizontal_thrust - drag * (self.speed / total_speed if total_speed else 0)
        horizontal_acceleration = horizontal_net_force / self.mass
        self.speed += horizontal_acceleration * timediff

        # Calculate lift and net vertical forces
        effective_lift_coefficient = self.lift_coefficient * math.cos(pitch_radians)
        lift = effective_lift_coefficient * (total_speed ** 2)
        net_vertical_force = lift + vertical_thrust - self.mass * self.gravity
        vertical_acceleration = net_vertical_force / self.mass
        self.vertical_speed += vertical_acceleration * timediff

    def simple_speed(self, timediff):
        '''Uses simplified forces to be more understandable'''
        #engine power
        thrust = self.accel * self.throttle
        horz_thrust = thrust * math.cos(math.radians(self.pitch_angle))
        vert_thrust = - thrust * math.sin(math.radians(self.pitch_angle))

        #wing lift and drag
        vert_lift = 9.8 * math.cos(math.radians(self.pitch_angle)) #simple lift, when wing points flat to ground it is about as strong as gravity
       
        #drag to resist motion and non aerodynamic faces
        drag_c = 1.2
        horz_drag = drag_c * self.speed * math.sin(math.radians(self.pitch_angle)) * timediff
        vert_drag = drag_c * self.vertical_speed * math.cos(math.radians(self.pitch_angle)) * timediff

        #speeds
        self.speed = self.speed + ( horz_thrust - horz_drag) * timediff                                     #horizontal
        self.vertical_speed = self.vertical_speed + (vert_thrust - 9.8 + vert_lift - vert_drag) * timediff  #vertical


    def calculate_altitude(self, timediff):
        #NOTE: Pygame altitudes are switched
        self.altitude -= self.vertical_speed * timediff

    def controls(self, throttle, elevator_angle):
        #throttle limits
        if(throttle > 0.5):
            throttle = 1
        elif( throttle < 0.5):
            throttle = 0
        self.throttle = throttle

        #elevator flap limits
        if(elevator_angle > 0.5):
            elevator_angle = 20
        elif( elevator_angle < 0.5):
            elevator_angle = -20
        self.elevator_angle = elevator_angle

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

def optimal_height(ap_alt, ap_x,heights,optimal = 500):
    #gives difference between optimal height and current flight height from the ground
    avg_mt_height = sum(heights[ap_x:(ap_x+10)])/10
    avg_ap_alt = math.fabs(ap_alt - avg_mt_height)
    return math.fabs(optimal - avg_ap_alt)

def plane_vectorize(plane, environment):
    '''
    Represents the airplane state as a vector of values
    '''

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


# class NetWithoutDropout(nn.Module):
#     def __init__(self,layer_dims):
#         '''
#         Builds a 4 layer deep network

#         args: 
#         layer_dims - list of nueron amounts at each layer

#         '''
#         super().__init__()
#         self.model_stack = nn.Sequential(
#             #4 layer network
#             nn.Linear(layer_dims[0],layer_dims[1]),
#             nn.ReLU(),
#             nn.Linear(layer_dims[1],layer_dims[2]),
#             nn.ReLU(),
#             nn.Linear(layer_dims[2],layer_dims[3]),
#             nn.ReLU(),
#             nn.Linear(layer_dims[3],layer_dims[4]),
#             nn.Sigmoid()
#         )

#     def forward(self,x):
#         y_hat = self.model_stack(x)
#         #layer prediction
#         return y_hat 
    
# # Weight initialization
# def init_weights(m):
#     '''Glorot weights'''
#     if isinstance(m, nn.Linear):
#         # Calculate the bounds for Glorot initialization
#         bound = np.sqrt(6 / (m.weight.size(0) + m.weight.size(1)))
#         # Apply Glorot initialization to the weights
#         torch.nn.init.uniform_(m.weight, -bound, bound)
#         # Initialize biases to zeros
#         m.bias.data.fill_(0.00)

def main():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Smooth Mountain Variation")

    clock = pygame.time.Clock()
    #1 second = 60 ticks
    #1 meter = 60 points
    tickspeed = 30  
    experiences = []

    # Initialize amplitude values for each segment
    amps = [65, 80, 20, 200, 130, 300, 40, 234, 100, 50, 78]

    game_over = False
    tick = 0    #tracks distances traveled in sets of 60
    mountain_points = generate_mountain_points(amps)

    ticks_per_meter = 60
    ticks_per_sec = 60
    
    # scoring
    distance_traveled = 0
    collision = False
    debug = False
    score = 0
    prev_speed = 65
    prev_angle = -1
    time = 0

    plane = Airplane(altitude=100, speed=prev_speed, pitch_angle=prev_angle)
    airplane_x = 50 #draw locations

    environment = (
        mountain_points[0],
        mountain_points[1],
        mountain_points[2]
        )
    airplane_vec = plane_vectorize(plane,environment)

    dims = [len(airplane_vec),51,25,15,2]
    # model = NetWithoutDropout(dims)
    # model.apply(init_weights)

    # model = DQN_agent(dims)
    # model.apply(model.init_weights)
    plane_rl = Plane_rl(dims)
    ep = 0

    while not game_over:
        

        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True

        # if collision: 
        #     display_message(screen, "Collision Detected!", RED, 50, HEIGHT // 2)
        #     clock.tick(tickspeed)
        #     continue

        # Generate mountain points with smooth variation
        # points = generate_mountain_points(amps)
        # mountain_points = mountain_points[1:] + [mountain_points[0]]  # Remove leftmost point


        ## Airplane
        plane.update(1/ticks_per_sec)
        environment = (
            mountain_points[0],
            mountain_points[1],
            mountain_points[2]
        )
        airplane_vec_cur = plane_vectorize(plane, environment)
        #add three height points
        
        # output = model(torch.FloatTensor(airplane_vec))
        # plane.controls(throttle=output[0],elevator_angle=output[1])
        output = plane_rl.get_action(ep, airplane_vec_cur)
        # plane.controls(throttle=1,elevator_angle=-4) #need to work on this
        plane.controls(output[0],output[1])

        plane.update(1/ticks_per_sec)
        environment = (
            mountain_points[0],
            mountain_points[1],
            mountain_points[2]
        )
        airplane_vec_next = plane_vectorize(plane, environment)

        # experiences.append((airplane_vec_cur, outputs, score, airplane_vec_next, failed))
        # pattern_set = plane_rl.generate_pattern_set(experiences)
        

        d_dist = plane.speed
        # d_dist = 1  #plane speed

        #prints every 60 ticks, meaning one tick every second on a tickspeed of 60
        #can be used to measure distance
        
        if debug or (tick !=0 and tick % 20 == 0):
            print("elevation: ", mountain_points[airplane_x] - plane.altitude)
            print("distance travelled: ", round(distance_traveled))
            print("airplane height: ", HEIGHT + plane.altitude)
            print("airplane speed: ", plane.speed)
            print("airplane pitch: ", plane.pitch_angle)
            print("Score: ", score)
    
            # print()
            state_b, target_q_values = plane_rl.generate_pattern_set(experiences)
            combined_data = list(zip(state_b, target_q_values))
            loss = plane_rl.train(combined_data)
            ep += 1

        #score metrics
        total_speed = math.sqrt(plane.speed ** 2 + plane.vertical_speed ** 2)
        # penalize change in values, 
        # priority to smooth flight, 
        # give one point for not crashing
        # give points for staying within the optimal height
        # oldscore = score
        # score = score - math.sqrt(prev_speed**2 + total_speed**2) - math.sqrt(prev_angle**2 + plane.pitch_angle**2) + (d_dist / ticks_per_meter) + 1 - optimal_height(plane.altitude,airplane_x,mountain_points, optimal=100)
        score = - math.sqrt(prev_speed**2 + total_speed**2) - math.sqrt(prev_angle**2 + plane.pitch_angle**2) + optimal_height(plane.altitude,airplane_x,mountain_points, optimal=100)

        # score_diff = oldscore - score
        
        #backpropagate
        # criterion = nn.BCELoss()
        # optimizer = optim.SGD(model.parameters(), lr=0.01)
    


        


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
        experiences.append((airplane_vec_cur, output, score, airplane_vec_next, collision))      

        if collision: 

            if torch.is_tensor(score):
                score = score.item()
            if torch.is_tensor(distance_traveled):
                distance_traveled = distance_traveled.item()

            # display_message(screen, "Collision Detected!", RED, 50, HEIGHT // 2)
            # display_message(screen, f"Score: {round(score + distance_traveled,ndigits=2)}", WHITE, WIDTH // 2, 20)

        pygame.display.flip()
        clock.tick(tickspeed) #frame speed



        

    pygame.quit()

if __name__ == "__main__":
    main()