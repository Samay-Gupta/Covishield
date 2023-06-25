from .static_data import RL_WIDTH
from math import sqrt
import geocoder
import datetime
import random

def get_center(rect):
    return (rect[0] + rect[2]//2, rect[1] + rect[3]//2)

def get_distance(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return sqrt((x2-x1)**2 + (y2-y1)**2)

def calculate_social_dist(rect1, rect2):
    pos1 = get_center(rect1)
    pos2 = get_center(rect2)
    pixel_dist = get_distance(pos1, pos2)
    act_width = RL_WIDTH
    rel_width = rect1[2]
    act_dist = pixel_dist*(act_width/rel_width)
    return act_dist

def get_location():
    mp = lambda x: [i + random.random() * random.choice([-1, 1]) * 0.0000000001 for i in x]
    return mp(geocoder.ip('me').latlng)

def get_date_and_time():
    now = datetime.datetime.now()
    return now.strftime("%m/%d/%Y"), now.strftime("%H:%M:%S")
