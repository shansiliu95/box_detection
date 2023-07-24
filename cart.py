from email.charset import add_charset
from utils.metrics import overlap_area
from collections import deque
import copy
import numpy as np
import time
import json
import random
import string
import requests

class Cart:
    def __init__(self, id, target_box, target_area, window_len=10, in_window_threshold=0.7, in_x=1, in_y=1):
        self.box_history = []
        self.overlap_ratio_history = []
        self.overlap_ratio_diff = []
        self.direction_history = []
        self.target_box = target_box
        self.target_area = target_area
        self.id = id
        self.window_lenth = window_len
        self.in_window_threshold = in_window_threshold
        self.in_x = in_x
        self.in_y = in_y
        self.last_update_time = time.time()
    
    def add_to_history(self, new_box):
        # assume box as [x_min, y_min, x_max, y_max ,probability, class]
        self.box_history.append(new_box[:4])
        if len(self.box_history) > 1:
            dx = (self.box_history[-1][0] + self.box_history[-1][2]).item() / 2 - (self.box_history[-2][0] + self.box_history[-2][2]).item() / 2
            dy = (self.box_history[-1][1] + self.box_history[-1][3]).item() / 2 - (self.box_history[-2][1] + self.box_history[-2][3]).item() / 2
            self.direction_history.append([dx, dy])
        self.overlap_ratio_history.append(overlap_area(new_box[:4], self.target_box) / self.target_area)
        if len(self.overlap_ratio_history) > 1:
            self.overlap_ratio_diff.append(self.overlap_ratio_history[-1] - self.overlap_ratio_history[-2])

        if len(self.overlap_ratio_diff) > self.window_lenth:
            self.box_history.pop(0)
            self.overlap_ratio_diff.pop(0)
            self.overlap_ratio_history.pop(0)
            self.direction_history.pop(0)

        self.last_update_time = time.time()
        

    def get_last_box(self):
        return self.box_history[-1]
    
    def get_avg_history_direction(self):
        past_direction = np.array(self.direction_history).mean(0)
        return past_direction[0], past_direction[1]
    
    def is_get_in(self):
        if len(self.overlap_ratio_diff) < self.window_lenth:
            return False
        is_increasing = np.mean(np.array(self.overlap_ratio_diff) > 0) > self.in_window_threshold
        past_dx, past_dy = self.get_avg_history_direction()
        correct_direction = past_dx * self.in_x >= 0 and past_dy * self.in_y >= 0
        return is_increasing and correct_direction
    
    def is_get_out(self):
        if len(self.overlap_ratio_diff) < self.window_lenth:
            return False
        is_decreasing = np.mean(np.array(self.overlap_ratio_diff) < 0) > self.in_window_threshold
        past_dx, past_dy = self.get_avg_history_direction()
        correct_direction = past_dx * self.in_x <= 0 and past_dy * self.in_y <= 0 
        return is_decreasing and correct_direction 

class ManyCart:
    def __init__(self, same_cart_threshold, target_box, target_area, window_len, in_window_threshold, in_x, in_y, obsolete_time, source):
        self.all_carts = []
        self.same_cart_threshold = same_cart_threshold
        self.target_box = target_box
        self.target_area = target_area
        self.window_len = window_len
        self.in_window_threshold = in_window_threshold
        self.num_in = 0
        self.num_out = 0
        self.in_counted_cart_id = set()
        self.out_counted_cart_id = set()
        self.in_x = in_x
        self.in_y = in_y
        self.obsolete_time = obsolete_time
        self.source = source

    def post_josn(self, type, count):

        results = dict()
        results['DOOR_ID'] = type
        results['CAM_ID'] = count
        results['INPUT'] = "2323"
        results['OUTPUT'] = self.source
        results['TIME'] = time.asctime(time.localtime())
        json_object = json.dumps(results, indent=4)
        url = 'http://192.168.1.200'
        headers = {'Content-type': 'application/json'}   
        response = requests.post(url, data=json_object, headers=headers)
        if response.status_code == 200:
            response_data = json.loads(response.content)
            print(response_data)
        else:
            print('Error:', response.status_code)

    def save_json(self, type, count):
        results = dict()
        results['type'] = type
        results['count'] = count
        results['time'] = time.asctime(time.localtime())
        results['source'] = self.source
        json_object = json.dumps(results, indent=4)
        rand_str = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=6))
        filename = results['time'].replace(" ", "_").replace(":", "_")
        with open(f"./json_files/{filename}_{rand_str}.json", "w") as outfile:
            outfile.write(json_object)

    def add_cart(self, new_box):
        new_cart = Cart(len(self.all_carts), self.target_box, self.target_area, self.window_len, self.in_window_threshold, self.in_x, self.in_y)
        new_cart.add_to_history(copy.deepcopy(new_box))
        self.all_carts.append(new_cart)

    def update_cart(self, new_box):
        new_area = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])
        max_area = -1
        if len(self.all_carts) == 0:
            self.add_cart(new_box)
            return
        for i, cart in enumerate(self.all_carts):
            last_box = cart.get_last_box()
            area = overlap_area(new_box[:4], last_box) 
            if area > max_area:
                max_area = area
                best_cart = i
        if max_area / new_area > self.same_cart_threshold:
            self.all_carts[best_cart].add_to_history(copy.deepcopy(new_box))
            if not best_cart in self.in_counted_cart_id:
                if self.all_carts[best_cart].is_get_in():
                    self.save_json('out', 1)
                    self.num_in += 1
                    self.in_counted_cart_id.add(best_cart)
            if not best_cart in self.out_counted_cart_id:
                if self.all_carts[best_cart].is_get_out():
                    self.save_json('in', 1)
                    self.num_out += 1
                    self.out_counted_cart_id.add(best_cart)
        else:
            self.add_cart(new_box)

    def clean_obsolete_cart(self):
        self.new_all_carts = []
        for i, cart in enumerate(self.all_carts):
            if time.time() - cart.last_update_time < self.obsolete_time:
                self.new_all_carts.append(cart)
        self.all_carts = self.new_all_carts[:]
        
        



