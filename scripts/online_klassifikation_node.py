#!/usr/bin/env python3

import rospy
import joblib
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import TwistStamped, WrenchStamped
import os
import re
import pandas as pd
from collections import deque
import csv
from datetime import datetime
from scipy.signal import correlate
from scipy.stats import entropy
from scipy.fft import fft, fftfreq
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import yaml
from message_filters import ApproximateTimeSynchronizer, Subscriber
from std_msgs.msg import Bool
import threading

class FileEventHandler(FileSystemEventHandler):
    def __init__(self, reload_callback, cooldown=2):
        self.reload_callback = reload_callback
        self.last_modified = 0
        self.cooldown = cooldown

    def on_modified(self, event):
        if event.src_path.endswith("Aktualisierung.yaml"):
            now = time.time()
            if now - self.last_modified > self.cooldown:
                self.last_modified = now
                rospy.loginfo(f"[INFO] YAML file updated: {event.src_path}, reloading parameters...")
                self.reload_callback()

            
class Klassifikation_Node:
    def __init__(self):
        rospy.init_node('online_klassifikation')
        
        self.base_path = rospy.get_param("~base_path", "/home/rtliu/catkin_ws/src/online_klassifikation_pkg/Best_Model/") # hier muss mit realen ordner entsprechen!!!
        self.model_type = None
        self.minSminL = None
        self.window_size = None
        self.split_method = None

        self.feature_call_count = 1
        self.processed_count = 1
        self.slow_process_count = 0 
        self.load_parameters()

        self.lock = threading.Lock()  
        self.thread = None

        self.model_path = None
        self.model = None
        
        self.pub_model_loaded = rospy.Publisher('/model_loaded', Bool, queue_size=1)
        rospy.sleep(1)

        self.update_model_path()
        self.load_model()
        self.old_params = (self.model_type, self.minSminL, self.window_size, self.split_method)
        self.data_window = deque(maxlen=self.window_size)

        self.twist_sub = Subscriber('/cart_vel', TwistStamped)
        self.wrench_sub = Subscriber('/wrench', WrenchStamped)
        
        self.ts = ApproximateTimeSynchronizer([self.twist_sub, self.wrench_sub], queue_size=10, slop=0.001)
        self.ts.registerCallback(self.sync_callback)



        self.result_publisher = rospy.Publisher('/classification_result', String, queue_size=100)
        
        rospy.Timer(rospy.Duration(5), self.check_model_update)
        self.start_config_watcher()

    def sync_callback(self, twist_msg, wrench_msg):

        velocity_data = [twist_msg.twist.linear.x, twist_msg.twist.linear.y, twist_msg.twist.linear.z]
        force_torque_data = [
            wrench_msg.wrench.force.x, wrench_msg.wrench.force.y, wrench_msg.wrench.force.z,
            wrench_msg.wrench.torque.x, wrench_msg.wrench.torque.y, wrench_msg.wrench.torque.z
        ]
        combined_data = velocity_data + force_torque_data
        self.process_data(combined_data)

    def load_parameters(self):
        yaml_path = "/home/rtliu/catkin_ws/src/online_klassifikation_pkg/scripts/Aktualisierung.yaml"

        with open(yaml_path, 'r') as file:
            try:
                params = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                rospy.logerr(f"Failed to load YAML: {exc}")
                return

        if params:
            self.model_type = params.get('model_type', self.model_type)
            self.minSminL = params.get('minSminL', self.minSminL)
            self.window_size = params.get('window_size', self.window_size)
            self.split_method = params.get('split_method', self.split_method)

    def start_config_watcher(self):
        event_handler = FileEventHandler(self.reload_callback)
        observer = Observer()
        observer.schedule(event_handler, path="/home/rtliu/catkin_ws/src/online_klassifikation_pkg/scripts/", recursive=False)
        observer.start()

    def reload_callback(self):
        rospy.loginfo("Reloading parameters due to YAML update...")
        self.check_model_update(None)


# model finden
    def find_model(self, model_dir):
        best_model = None

        try:
            files = os.listdir(model_dir)

            if self.model_type == "SVM":
                # SVM ohne depth
                pattern = re.compile(f"^{self.split_method}\.pkl$")
            else:
                # Decision tree and ranfom forest
                pattern = re.compile(f"^{self.split_method}_depth_(\d+)\.pkl$")

            for file in files:
                match = pattern.match(file)
                if match:
                    best_model = file

            return os.path.join(model_dir, best_model) if best_model else None

        except Exception as e:
            rospy.logerr(f"Error finding best model: {e}")
            return None

    def update_model_path(self):
        # neue Model aktualisieren
        if self.model_type == "SVM":
            model_dir = os.path.join(
                self.base_path,
                self.model_type,
                f"Window_{self.window_size}"
            )
        else:
            model_dir = os.path.join(
                self.base_path,
                self.model_type,
                self.minSminL,
                f"Window_{self.window_size}"
            )

        if not os.path.exists(model_dir):
            rospy.logerr(f"Model directory not found: {model_dir}")
            self.model_path = None
            return

        find_model_path = self.find_model(model_dir)

        if find_model_path:
            self.model_path = find_model_path
            rospy.loginfo(f"Found model: {self.model_path}")
        else:
            rospy.logerr(f"No matching model found in {model_dir} for {self.split_method}")

    def load_model(self):
        if self.model_path and os.path.exists(self.model_path):   
            self.model = joblib.load(self.model_path)
            rospy.loginfo(f"Loaded model: {self.model_path}")
            self.pub_model_loaded.publish(True)
        else:
            rospy.logerr(f"Model file {self.model_path} not found!")
            self.pub_model_loaded.publish(False)

            

    def check_model_update(self, event):
        rospy.loginfo("Checking if model parameters changed...")

        self.load_parameters()
    
        new_params = (self.model_type, self.minSminL, self.window_size, self.split_method)

        if self.old_params != new_params:
            rospy.loginfo(f"Model parameters changed: {self.old_params} → {new_params}")
            self.update_model_path()
            self.load_model()
            self.old_params = new_params
        else:
            rospy.logdebug("No changes detected.") 

# sub data, sliding window 
    # def velocity_callback(self, msg):
    #     self.velocity_buffer = [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z]
    
    # def wrench_callback(self, msg):
        
    #     force_torque_data = [
    #         msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
    #         msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
    #     ]
    #     if hasattr(self, "velocity_buffer"): 
    #         combined_data = self.velocity_buffer + force_torque_data
    #         self.process_data(combined_data)
    #         del self.velocity_buffer 
    #     self.wrench_count += 1
    #     rospy.loginfo(f"Callback triggered: {self.wrench_count}, Window size: {len(self.data_window)}")

    # def process_data(self, data):
    #     rospy.loginfo("Processing data...")
    #     start_time = time.time()
    
    #     self.data_window.append(data)
    #     rospy.loginfo(f"Current window size: {len(self.data_window)}")
        
    #     if len(self.data_window) == self.window_size:
    #         rospy.loginfo("Window size reached, calculating features...")
    #         self.calculate_features()
    #         rospy.loginfo(f"Processed data count: {self.processed_count}")
    #         self.processed_count += 1
    #     end_time = time.time()
    #     duration = end_time - start_time

    #     if duration > 0.01:
    #         self.slow_process_count += 1
    #     rospy.loginfo(f"duration: {duration}s")
    #     rospy.loginfo(f"Slow count: {self.slow_process_count}")
    def process_data(self, data):
        start_time = time.time()

        with self.lock:
            self.data_window.append(data)

        if len(self.data_window) == self.window_size:
            rospy.loginfo("Submitting feature extraction to background thread...")
            self.thread = threading.Thread(target=self.calculate_features)
            self.thread.start()

        duration = time.time() - start_time
        if duration > 0.01:
            self.slow_process_count += 1
        rospy.loginfo(f"duration: {duration}s")
        rospy.loginfo(f"Slow count: {self.slow_process_count}")

    @staticmethod
    def autocorrelation(series):
        n = len(series)
        result = correlate(series, series, mode='full') / n
        mid_point = len(result) // 2
        return result[mid_point]

    @staticmethod
    def hjorth_complexity(series):
        if np.std(series) == 0:
            return 0
        diff_series = np.diff(series)
        if np.std(diff_series) == 0:
            return 0
        diff2_series = np.diff(diff_series)
        mobility = np.std(diff_series) / np.std(series)
        complexity = (np.std(diff2_series) / np.std(diff_series)) / mobility if mobility != 0 else 0
        return complexity if not np.isnan(complexity) else 0

    @staticmethod
    def shannon_entropy(series):
        prob_distribution, _ = np.histogram(series, bins=10, density=True)
        prob_distribution = prob_distribution[prob_distribution > 0]  # 过滤 0 概率
        return entropy(prob_distribution) if len(prob_distribution) > 0 else 0

    @staticmethod
    def dominant_frequency(series, sampling_rate=100):
        n = len(series)
        if n < 2:
            return 0
        yf = fft(series)
        xf = fftfreq(n, 1 / sampling_rate)[:n // 2]
        magnitude = 2.0 / n * np.abs(yf[:n // 2])
        return xf[np.argmax(magnitude)] if len(magnitude) > 0 else 0

    # def calculate_features(self):
    #     data = pd.DataFrame(self.data_window, columns=['v_x', 'v_y', 'v_z', 'F_x', 'F_y', 'F_z', 'M_x', 'M_y', 'M_z'])
    #     features = np.concatenate([
    #         data.mean().values, 
    #         data.std().values,
    #         data.max().values, 
    #         data.min().values, 
    #         data.apply(self.autocorrelation, raw=True).values,
    #         data.apply(self.hjorth_complexity, raw=True).values,
    #         data.apply(self.shannon_entropy, raw=True).values,
    #         data.apply(self.dominant_frequency, raw=True).values
    #     ]).reshape(1, -1)
    #     # print("Features:\n", features)
    #     rospy.loginfo(f"Feature extraction called {self.feature_call_count} times")
    #     self.feature_call_count += 1
    #     self.classify(features)

    def calculate_features(self):
        data_array = np.array(self.data_window)  # 转换为 NumPy 数组
        mean_features = np.mean(data_array, axis=0)
        std_features = np.std(data_array, axis=0)
        max_features = np.max(data_array, axis=0)
        min_features = np.min(data_array, axis=0)

        autocorr_features = np.apply_along_axis(self.autocorrelation, axis=0, arr=data_array)
        hjorth_features = np.apply_along_axis(self.hjorth_complexity, axis=0, arr=data_array)
        entropy_features = np.apply_along_axis(self.shannon_entropy, axis=0, arr=data_array)
        dom_freq_features = np.apply_along_axis(self.dominant_frequency, axis=0, arr=data_array)

        # 拼接所有特征
        feature_vector = np.concatenate([
            mean_features, std_features, max_features, min_features,
            autocorr_features, hjorth_features, entropy_features, dom_freq_features
        ]).reshape(1, -1)

        rospy.loginfo(f"Feature extraction called {self.feature_call_count} times")
        self.feature_call_count += 1
        self.classify(feature_vector)  # 调用分类器


    # def calculate_features(self):
    #     start_time = time.time()  # 记录计算开始时间

    #     data = np.array(self.data_window)  # 直接转 NumPy 数组，加速计算
    #     mean_vals = np.mean(data, axis=0)
    #     std_vals = np.std(data, axis=0)
    #     max_vals = np.max(data, axis=0)
    #     min_vals = np.min(data, axis=0)

        
    #     auto_corr = np.array([np.correlate(data[:, i], data[:, i], mode='full')[len(data) - 1] / len(data) for i in range(data.shape[1])])

    #     diff_data = np.diff(data, axis=0)
    #     diff2_data = np.diff(diff_data, axis=0)
    #     mobility = np.std(diff_data, axis=0) / np.std(data, axis=0)
    #     complexity = (np.std(diff2_data, axis=0) / np.std(diff_data, axis=0)) / mobility
    #     complexity[np.isnan(complexity)] = 0  # 避免 NaN

    #     hist, _ = np.histogram(data, bins=10, density=True)
    #     hist = hist[hist > 0]
    #     entropy_vals = -np.sum(hist * np.log2(hist))

    #     n = data.shape[0]
    #     fft_vals = np.abs(np.fft.rfft(data, axis=0))  # 只取正频率部分
    #     dom_freq = np.argmax(fft_vals, axis=0) / n * 100  # 计算主频率

    #     features = np.concatenate([mean_vals, std_vals, max_vals, min_vals, auto_corr, complexity, [entropy_vals], dom_freq]).reshape(1, -1)

    #     rospy.loginfo(f"Feature extraction called {self.feature_call_count} times")
    #     self.feature_call_count += 1

    #     self.classify(features)

    #     end_time = time.time()
    #     rospy.loginfo(f"Feature extraction duration: {end_time - start_time:.6f}s")


    def classify(self, features):
        class_mapping = {0: "No Contact", 1: "Interaction", 2: "Collision"}
        if self.model:
            prediction = self.model.predict(features)[0]
            
            result_text = class_mapping.get(prediction, "Unknown")  # 兜底防止意外值
            result_value = prediction

            save_dir = "/home/rtliu/catkin_ws/src/online_klassifikation_pkg/classification_results/"
            os.makedirs(save_dir, exist_ok=True)  
            csv_file = os.path.join(save_dir, "classification_results.csv")

            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)

                if os.stat(csv_file).st_size == 0:
                    writer.writerow(["Classification", "Class_Label"])

                writer.writerow([result_text, result_value])

            self.result_publisher.publish(result_text)

            rospy.loginfo(f"Classification result saved: {result_text} ({result_value})")


if __name__ == "__main__":
    node = Klassifikation_Node()
    rospy.spin()





