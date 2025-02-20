#!/usr/bin/env python3

import rospy
import joblib
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import TwistStamped, WrenchStamped
import os
import re
from collections import deque
import csv
from scipy.fft import fft, fftfreq
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import yaml
from message_filters import ApproximateTimeSynchronizer, Subscriber
from std_msgs.msg import Bool
from line_profiler import LineProfiler


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


''' Achtung: 
    Before each use, 
    please ensure that the <!-- Ordnerpfad orientieren --> section 
    in the classification_simulation.launch file is correctly configured. '''

class Klassifikation_Node:
    def __init__(self):
        rospy.init_node('online_klassifikation')
        
        self.yaml_path = rospy.get_param("~yaml_path")
        self.observe_path = rospy.get_param("~observe_path")

        self.save_dir = rospy.get_param("~save_dir")
        self.csv_file = os.path.join(self.save_dir, "classification_results.csv")

        # 初始化时检查 CSV 文件是否存在，不存在则写入表头
        os.makedirs(self.save_dir, exist_ok=True)
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Classification", "Class_Label"]) 


        self.base_path = rospy.get_param("~base_path", "/default/path/if/not/set")
        self.model_type = None
        self.minSminL = None
        self.window_size = None
        self.split_method = None
        self.load_parameters()

        self.data_window = deque(maxlen=self.window_size)
        self.next_prediction = None


        self.slow_process_count = 0
        self.processed_count = 1
        
        self.pub_model_loaded = rospy.Publisher('/model_loaded', Bool, queue_size=1) 
        rospy.sleep(1)  

        self.model_path = None
        self.model = None
        self.scaler = None
        self.update_model_path()
        self.load_model()
        self.old_params = (self.model_type, self.minSminL, self.window_size, self.split_method)

        # subscribe the sensor data
        self.twist_sub = Subscriber('/cart_vel', TwistStamped)
        self.wrench_sub = Subscriber('/wrench', WrenchStamped)
        self.ts = ApproximateTimeSynchronizer([self.twist_sub, self.wrench_sub], queue_size=10, slop=0.001)
        self.ts.registerCallback(self.sync_callback)

        # publish the classification result
        self.result_publisher = rospy.Publisher('/classification_result', String, queue_size=100)

        # check the update of the predict model
        rospy.Timer(rospy.Duration(5), self.check_model_update)
        self.start_config_watcher()

    ''' This part is Input Processing '''
    # Input synchronization
    def sync_callback(self, twist_msg, wrench_msg):

        velocity_data = [twist_msg.twist.linear.x, twist_msg.twist.linear.y, twist_msg.twist.linear.z]
        force_torque_data = [
            wrench_msg.wrench.force.x, wrench_msg.wrench.force.y, wrench_msg.wrench.force.z,
            wrench_msg.wrench.torque.x, wrench_msg.wrench.torque.y, wrench_msg.wrench.torque.z
        ]
        combined_data = velocity_data + force_torque_data
        self.process_data(combined_data)

    def load_parameters(self):

        try:
            with open(self.yaml_path, 'r') as file:
                params = yaml.safe_load(file)
                if params:
                    self.model_type = params.get('model_type', self.model_type)
                    self.minSminL = params.get('minSminL', self.minSminL)
                    self.window_size = params.get('window_size', self.window_size)
                    self.split_method = params.get('split_method', self.split_method)
        except yaml.YAMLError as exc:
            rospy.logerr(f"Failed to load YAML: {exc}")

    def start_config_watcher(self):
        event_handler = FileEventHandler(self.reload_callback)
        observer = Observer()
        observer.schedule(event_handler, self.observe_path, recursive=False)
        observer.start()

    def reload_callback(self):
        rospy.loginfo("Reloading parameters due to YAML update...")
        self.check_model_update(None)

    ''' This part is model lookup, update and load '''
    # model finden
    def find_model(self, model_dir):
        try:
            files = os.listdir(model_dir)
            pattern = re.compile(f"^{self.split_method}\.pkl$") if self.model_type == "SVM" else re.compile(f"^{self.split_method}_depth_(\d+)\.pkl$")
            for file in files:
                if pattern.match(file):
                    return os.path.join(model_dir, file)
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
            if self.model_type == "SVM" and self.split_method:
                # 根据 split_method 构造 scaler 文件路径
                scaler_path = os.path.join(os.path.dirname(self.model_path), f"scaler_{self.split_method}.pkl")
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                    rospy.loginfo(f"Loaded scaler: {scaler_path}")
                else:
                    rospy.logwarn(f"Scaler file {scaler_path} not found!")

            if hasattr(self.model, "n_jobs"):  
                self.model.n_jobs = 1
            rospy.loginfo(f"Loaded model: {self.model_path}")
            self.pub_model_loaded.publish(True)
        else:
            rospy.logerr(f"Model file {self.model_path} not found!")
            self.pub_model_loaded.publish(False)

    def check_model_update(self, event):

        self.load_parameters()
        new_params = (self.model_type, self.minSminL, self.window_size, self.split_method)
        if self.old_params != new_params:
            rospy.loginfo(f"Model parameters changed: {self.old_params} → {new_params}")
            self.update_model_path()
            self.load_model()
            self.old_params = new_params
    
    ''' Calculate code processing speed line by line '''
    def profile_line_by_line(func):
        def wrapper(*args, **kwargs):
            profiler = LineProfiler()
            profiler.add_function(func)
            profiler.enable_by_count()
            result = func(*args, **kwargs)
            profiler.disable()
            profiler.print_stats(output_unit=1e-3)
            return result
        return wrapper
    
    ''' Processing part: includes feature calculation, classification, and result saving. '''
    @profile_line_by_line 
    def process_data(self, data):
        
        start_time = time.time()
        self.data_window.append(data)

        if len(self.data_window) == self.window_size:
            # self.executor.submit(self.calculate_features)
            self.calculate_features()
            self.processed_count += 1
            # rospy.loginfo(f"Processed data count: {self.processed_count}")
        duration = time.time() - start_time
        if duration > 0.01:
            self.slow_process_count += 1
        rospy.loginfo(f"Processed data count: {self.processed_count}, Slow count: {self.slow_process_count}")

    @staticmethod
    def autocorrelation_matrix(data_array):
        return np.array([
            np.correlate(col, col, mode='full')[len(col) // 2] / len(col)
            for col in data_array.T
        ])

    def hjorth_complexity_matrix(self, data_array):
        std_series = np.std(data_array, axis=0)
        diff_series = np.diff(data_array, axis=0)
        std_diff = np.std(diff_series, axis=0)
        diff2_series = np.diff(diff_series, axis=0)
        std_diff2 = np.std(diff2_series, axis=0)

        mobility = np.where(std_series != 0, std_diff / std_series, 0)
        complexity = np.where((mobility != 0) & (std_diff != 0), (std_diff2 / std_diff) / mobility, 0)

        return np.nan_to_num(complexity)

    @staticmethod
    def shannon_entropy_matrix(data_array, bins=5):
        num_columns = data_array.shape[1]  
        entropy_values = np.zeros(num_columns) 

        for i in range(num_columns):
            hist_data, _ = np.histogram(data_array[:, i], bins=bins, density=True)  
            hist_data = hist_data / np.sum(hist_data) 

            hist_data[hist_data == 0] = 1 
            entropy_values[i] = -np.sum(hist_data * np.log2(hist_data))  

        return entropy_values

    @staticmethod
    def dominant_frequency_matrix(data_array, sampling_rate=100):
        n = data_array.shape[0]
        yf = fft(data_array, axis=0)
        xf = fftfreq(n, 1 / sampling_rate)[:n // 2]
        magnitude = 2.0 / n * np.abs(yf[:n // 2])
        return xf[np.argmax(magnitude, axis=0)] 

    # @profile_line_by_line
    def calculate_features(self):

        data_array = np.array(self.data_window)  # NumPy 
        mean_features = np.mean(data_array, axis=0)
        std_features = np.std(data_array, axis=0)
        max_features = np.max(data_array, axis=0)
        min_features = np.min(data_array, axis=0)

        autocorr_features = self.__class__.autocorrelation_matrix(data_array)
        hjorth_features = self.hjorth_complexity_matrix(data_array) 
        entropy_features = self.__class__.shannon_entropy_matrix(data_array)
        dom_freq_features = self.__class__.dominant_frequency_matrix(data_array)

        feature_vector = np.hstack([mean_features, std_features, max_features, min_features,
                                    autocorr_features, hjorth_features, entropy_features, dom_freq_features])
        if self.model_type == "SVM" and self.scaler:
            feature_vector = self.scaler.transform(feature_vector.reshape(1, -1)).flatten()
            # rospy.loginfo("Applied scaler transformation on feature vector.")

        self.classify(feature_vector)  

    # @profile_line_by_line
    def classify(self, feature_vector):

        class_mapping = {0: "No Contact", 1: "Interaction", 2: "Collision"}
        
        feature_vector = feature_vector.reshape(1, -1)  # Reshape to (1, n_features)
        
        if self.model:
            prediction = self.model.predict(feature_vector)[0]
            result_text = class_mapping.get(prediction, "Unknown")
            self.save_result(result_text, prediction)
        # rospy.loginfo(f"class: {class_time}s")

    def save_result(self, result_text, prediction):
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([result_text, prediction])

if __name__ == "__main__":
    node = Klassifikation_Node() 
    rospy.spin()





