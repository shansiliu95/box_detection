from email.charset import add_charset
from utils.metrics import overlap_area
from collections import deque
import copy
import numpy as np
import time
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
        self.last_update_time = None
    
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
    def __init__(self, same_cart_threshold, target_box, target_area, window_len, in_window_threshold, in_x, in_y):
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
                        self.num_in += 1
                        self.in_counted_cart_id.add(best_cart)
            if not best_cart in self.out_counted_cart_id:
                if self.all_carts[best_cart].is_get_out():
                        self.num_out += 1
                        self.out_counted_cart_id.add(best_cart)
        else:
            self.add_cart(new_box)


