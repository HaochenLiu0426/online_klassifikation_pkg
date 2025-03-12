#!/usr/bin/env python3

import rospy
import joblib
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import TwistStamped, WrenchStamped
import os
import re
import csv
from scipy.fft import rfft, rfftfreq
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import yaml
from message_filters import ApproximateTimeSynchronizer, Subscriber
from std_msgs.msg import Bool
from line_profiler import LineProfiler
import threading
import queue
import subprocess
from threading import Lock
from threading import Thread

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
        self.clean_ros_logs()
        self.yaml_path = rospy.get_param("~yaml_path")
        self.observe_path = rospy.get_param("~observe_path")

        # Create new csv document
        self.save_dir = rospy.get_param("~save_dir")
        self.csv_file = os.path.join(self.save_dir, "classification_results.csv")
        os.makedirs(self.save_dir, exist_ok=True)
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Classification", "Class_Label"]) 

        self.base_path = rospy.get_param("~base_path", "/default/path/if/not/set")
        self.pub_model_loaded = rospy.Publisher('/model_loaded', Bool, queue_size=1) 
        rospy.sleep(1)  
        self.prediction_lock = Lock()
        # Hyperparameter
        self.model_type = None
        self.minSminL = None
        self.window_size = None
        self.split_method = None
        self.load_parameters()

        # Model relativ parameter
        self.model_path = None
        self.model = None
        self.scaler = None
        

        # Time and Count
        self.publish_rate = rospy.get_param("/simulation_node/publish_rate")  
        self.publish_interval = 1.0 / self.publish_rate 
        self.slow_process_count = 0
        self.processed_count = 1
        self.last_update_time = None


        self.start_saving = False  # 是否开始保存结果的标志
        self.last_data_time = None 
        
        # Threading and Synchronization Control
        # self.prediction_lock = threading.Lock()
        self.last_prediction = 0
        self.last_pred_text = "No Contact"
        self.lock = threading.Lock()
        self.buffer_lock = threading.Lock()
        # loading relativ variable, check the update of the predict model
        self.update_model_path()
        self.load_model()
        self.old_params = (self.model_type, self.minSminL, self.window_size, self.split_method)

        rospy.Timer(rospy.Duration(5), self.check_model_update)
        self.activity_timer = None


        # self.publish_timer = rospy.Timer(rospy.Duration(self.publish_interval), self.publish_result)
        
        # subscribe the sensor data, publish the classification result
        self.twist_sub = Subscriber('/cart_vel', TwistStamped)
        self.wrench_sub = Subscriber('/wrench', WrenchStamped)
        # self.ts = ApproximateTimeSynchronizer([self.twist_sub, self.wrench_sub], queue_size=200, slop=0.0006)
        self.slop = 0.3*self.publish_interval
        self.ts = ApproximateTimeSynchronizer([self.twist_sub, self.wrench_sub],queue_size=250, slop=self.slop)

        self.ts.registerCallback(self.sync_callback)
        self.result_publisher = rospy.Publisher('/classification_result', String, queue_size=10)
        
        self.data_active = True  # 标记数据流是否活跃
        self.stop_prediction = False

        self.data_window = np.zeros((self.window_size, 9), dtype=np.float32)  
        self.class_mapping = {0: "No Contact", 1: "Interaction", 2: "Collision"}
        self.write_pos = 0
        self.full = False
        self.old_data = None
        self.latest_data = None

        self.result_queue = queue.Queue()

        self.running = True
        self.writer_thread = threading.Thread(target=self.save_result, daemon=True)
        self.writer_thread.start()


        self.result_text = 'No contact'
        self.prediction = 0
        self.pub_count = 0
        self.overwrite_count = 0 
        self.update_count = 0
        self.process_running = threading.Event()  # ✅ 事件标志，控制是否有线程在运行
        self.monitor_timer = None
        self.start_config_watcher()

    def clean_ros_logs(self):
        try:
            subprocess.run(["rosclean", "purge", "-y"], check=True)
            print("✅ ROS logs cleared successfully.")
        except Exception as e:
            print(f"⚠️ Failed to clear ROS logs: {e}")

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

    def append_to_buffer(self, data):
        """环形缓冲区写入数据"""
        with self.buffer_lock:
            self.data_window[self.write_pos] = data
            self.write_pos = (self.write_pos + 1) % self.window_size
            # self.overwrite_count += 1
            # rospy.logwarn(f"overwrite: {self.overwrite_count - 50}")
            if not self.full and self.write_pos == 0:
                self.full = True 
            # if self.full:  # 仅在缓冲区满时记录有效覆盖
            #     self.overwrite_count += 1
            #     rospy.logwarn(f"overwrite: {self.overwrite_count}")

    def get_window_copy(self):
        """获取当前的窗口数据"""
        with self.buffer_lock:
            if self.full:
                return np.concatenate((
                self.data_window[self.write_pos:],  # 先取 write_pos 后面的数据
                self.data_window[:self.write_pos]   # 再取 write_pos 前面的数据
            ), axis=0).copy()
            else:
                return self.data_window[:self.write_pos].copy()

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
            return

    def start_config_watcher(self):
        event_handler = FileEventHandler(self.reload_callback)
        observer = Observer()
        observer.schedule(event_handler, self.observe_path, recursive=False)
        observer.start()

    def reload_callback(self):
        rospy.loginfo("Reloading parameters due to YAML update...")
        self.check_model_update(None)


    ''' This part is model lookup, update and load '''
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

    
    ''' This part is Input Processing '''
    # Input synchronization
    # @profile_line_by_line
    def sync_callback(self, twist_msg, wrench_msg):
        velocity_data = np.array([
            twist_msg.twist.linear.x, twist_msg.twist.linear.y, twist_msg.twist.linear.z
        ], dtype=np.float32)

        force_torque_data = np.array([
            wrench_msg.wrench.force.x, wrench_msg.wrench.force.y, wrench_msg.wrench.force.z,
            wrench_msg.wrench.torque.x, wrench_msg.wrench.torque.y, wrench_msg.wrench.torque.z
        ], dtype=np.float32)

        self.latest_data = np.concatenate((velocity_data, force_torque_data))
        

        if self.activity_timer is None and self.monitor_timer is None:
            self.activity_timer = rospy.Timer(rospy.Duration(0.01), self.check_activity)
            self.monitor_timer = rospy.Timer(rospy.Duration(0.01), self.monitor_data_status)
        if self.full:
            self.publish_result()    

        self.update_count += 1
        if self.update_count >= int(self.publish_rate/100):  # 1000Hz / 10 = 100Hz
            self.fetch_and_store_data()
            self.update_count = 0
        

        self.last_update_time = time.time()

    def monitor_data_status(self, event):
        elapsed_time = time.time()- self.last_update_time

        if elapsed_time > 0.03:  # ✅ **超过 30ms 没有新数据，停止预测**
            self.data_active = False
            self.stop_prediction = True
            # self.check_activity(None)
            rospy.logwarn("Data timeout! Stopping prediction.")
                
    # @profile_line_by_line 
    # def fetch_and_store_data(self, event):   
    def fetch_and_store_data(self):    
        if not np.array_equal(self.old_data, self.latest_data):
            self.old_data = self.latest_data
            self.stop_prediction = False
            self.append_to_buffer(self.latest_data)

            if self.full and not self.stop_prediction:
                self.start_saving = True 
                if not self.process_running.is_set():  # 防止多个线程被创建
                    self.process_running.set()
                    threading.Thread(target=self.process_data, daemon=True).start()
                   
            
    ''' Processing part: includes feature calculation, classification, and result saving. '''
    # @profile_line_by_line 
    def process_data(self,event=None): 
        # with self.lock:
        while not self.stop_prediction:
            window_copy = self.get_window_copy()
            self.calculate_features(window_copy) 
        self.process_running.clear()
    
    @staticmethod
    def autocorrelation_matrix(data_array):
        return np.array([
            np.correlate(col, col, mode='full')[len(col) // 2] / len(col)
            for col in data_array.T
        ])

    def hjorth_complexity_matrix(self, data_array):
        epsilon = 1e-10  # 避免除零的小数值
        diff_series = np.diff(data_array, axis=0)
        diff2_series = np.diff(data_array, n=2, axis=0)

        std_series = np.std(data_array, axis=0)
        std_diff = np.std(diff_series, axis=0)
        std_diff2 = np.std(diff2_series, axis=0)

        # aviod 0 fehler
        mobility = np.where(std_series > epsilon, std_diff / (std_series + epsilon), 0)
        complexity = np.where(
            (mobility > epsilon) & (std_diff > epsilon),
            (std_diff2 / (std_diff + epsilon)) / (mobility + epsilon),
            0
        )
        # NaN and Inf
        complexity = np.nan_to_num(complexity)

        if np.any(np.isnan(complexity)) or np.any(np.isinf(complexity)):
            rospy.logwarn("Hjorth Complexity computation produced NaN or Inf values!")
        return complexity

    @staticmethod
    def shannon_entropy_matrix(data_array, bins=5):
        num_columns = data_array.shape[1]  
        entropy_values = np.zeros(num_columns) 

        for i in range(num_columns):
            hist_data, _ = np.histogram(data_array[:, i], bins=bins, density=True)  
            hist_data += 1e-10
            entropy_values[i] = -np.sum(hist_data * np.log2(hist_data))
        return entropy_values

    @staticmethod
    def dominant_frequency_matrix(data_array, sampling_rate=100):
        n = data_array.shape[0]
        yf = rfft(data_array, axis=0)  # 计算 FFT
        magnitude = np.abs(yf)  # 计算幅值
        freqs = rfftfreq(n, d=1/sampling_rate)  # 计算频率刻度
        return freqs[np.argmax(magnitude, axis=0)]

    # @profile_line_by_line
    def calculate_features(self, window_copy):
        mean_features = np.mean(window_copy, axis=0)
        std_features = np.std(window_copy, axis=0)
        max_features = np.max(window_copy, axis=0)
        min_features = np.min(window_copy, axis=0)
        autocorr_features = self.__class__.autocorrelation_matrix(window_copy)
        hjorth_features = self.hjorth_complexity_matrix(window_copy) 
        entropy_features = self.__class__.shannon_entropy_matrix(window_copy)
        dom_freq_features = self.__class__.dominant_frequency_matrix(window_copy)

        feature_vector = np.concatenate((
        mean_features, std_features, max_features, min_features,
        autocorr_features, hjorth_features, entropy_features, dom_freq_features
    ), dtype=np.float32)

        if self.model_type == "SVM" and self.scaler:
            feature_vector = self.scaler.transform(feature_vector.reshape(1, -1))[0]
        threading.Thread(target=self.classify, args=(feature_vector,), daemon=True).start()
        # self.classify(feature_vector)  

    # @profile_line_by_line
    def classify(self, feature_vector):
        feature_vector = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        prediction = self.model.predict(feature_vector)[0]
        result_text = self.class_mapping.get(prediction, "Unknown")
        
        # with self.prediction_lock:
        self.last_prediction = prediction
        self.last_pred_text = result_text
            
    # @profile_line_by_line
    def publish_result(self):
        self.pub_count += 1
        self.result_queue.put((self.last_pred_text, self.last_prediction))


    def save_result(self):
        while self.running or not self.result_queue.empty():  # 确保所有数据写入
            try:
                result_text, prediction = self.result_queue.get(timeout=0.005)
                with open(self.csv_file, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([result_text, prediction])
            except queue.Empty:
                if not self.running:  # 只有在 running=False 后，才检查是否要退出
                    break


    def check_activity(self, event):
        if not self.start_saving:
            return

        if not self.data_active and self.stop_prediction:
            rospy.loginfo("No new data, stopping prediction.")
            self.running = False
            # **停止 `publish_result`**
            # if self.publish_timer:
            #     rospy.loginfo("Stopping publish_result due to inactivity.")
            #     self.publish_timer.shutdown()
            #     # self.process_timer.shutdown()
            #     self.publish_timer = None

            # **停止 `activity_timer`**
            if self.activity_timer:
                rospy.loginfo("Stopping activity check timer.")
                self.activity_timer.shutdown()
                self.activity_timer = None

            if self.monitor_timer:
                self.monitor_timer.shutdown()
                self.monitor_timer = None

            # with self.result_queue.mutex:  # 线程安全操作
            #     self.result_queue.queue.clear()
            rospy.loginfo("Flushing remaining results before exit...")
            self.writer_thread.join()  # **等待 `save_result` 线程写完数据**
            rospy.loginfo("All data saved. Exiting.")
            rospy.loginfo(f"Pub_result count:{self.pub_count}")
if __name__ == "__main__":
    node = Klassifikation_Node() 
    rospy.spin()





