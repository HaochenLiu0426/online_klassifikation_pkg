#!/usr/bin/env python3

import rospy
import pandas as pd
import yaml
from std_msgs.msg import Bool
from geometry_msgs.msg import TwistStamped, WrenchStamped
import os

class SimulationNode:
    def __init__(self, csv_path):
        rospy.init_node('simulation_node')

        self.model_loaded = False
        rospy.Subscriber('/model_loaded', Bool, self.model_callback)

        # alte csv löschen
        results_path = "/home/rtliu/catkin_ws/src/online_klassifikation_pkg/classification_results/classification_results.csv"
        if os.path.exists(results_path):
            rospy.loginfo("Removing old classification_results.csv to avoid misalignment...")
            os.remove(results_path)

    
        # read csv
        self.data = pd.read_csv(csv_path, delimiter=";")
        self.index = 0  

        # create publisher
        self.twist_pub = rospy.Publisher('/cart_vel', TwistStamped, queue_size=10)
        self.wrench_pub = rospy.Publisher('/wrench', WrenchStamped, queue_size=10)

        # 1ms/ 10ms rate
        publish_rate = rospy.get_param("~publish_rate")
        self.rate = rospy.Rate(publish_rate) 

    def model_callback(self, msg):
        if msg.data:
            rospy.loginfo("model loaded, simulation_node start")
            self.model_loaded = True 

    def run(self):
        while not rospy.is_shutdown() and not self.model_loaded:
            rospy.sleep(0.1)  

        while not rospy.is_shutdown() and self.index < len(self.data):
            row = self.data.iloc[self.index]

            # vel
            twist_msg = TwistStamped()
            twist_msg.header.stamp = rospy.Time.now()
            twist_msg.twist.linear.x = row['v_x']
            twist_msg.twist.linear.y = row['v_y']
            twist_msg.twist.linear.z = row['v_z']

            # kraft moment
            wrench_msg = WrenchStamped()
            wrench_msg.header.stamp = rospy.Time.now()
            wrench_msg.wrench.force.x = row['F_x']
            wrench_msg.wrench.force.y = row['F_y']
            wrench_msg.wrench.force.z = row['F_z']
            wrench_msg.wrench.torque.x = row['M_x']
            wrench_msg.wrench.torque.y = row['M_y']
            wrench_msg.wrench.torque.z = row['M_z']

            # publish
            self.twist_pub.publish(twist_msg)
            self.wrench_pub.publish(wrench_msg)

            # protokol
            rospy.loginfo(f"Published data row {self.index}: Twist & Wrench")
            
            self.index += 1
            self.rate.sleep()

def load_csv_path(yaml_path):
    try:
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)
            return config.get("csv_path", None)
    except Exception as e:
        rospy.logerr(f"Error reading YAML file: {e}")
        return None

if __name__ == '__main__':
    try:
        yaml_file = "/home/rtliu/catkin_ws/src/online_klassifikation_pkg/scripts/Aktualisierung.yaml"
        csv_path = load_csv_path(yaml_file)

        if csv_path:
            rospy.loginfo(f"Loaded CSV path from YAML: {csv_path}")
            node = SimulationNode(csv_path)
            node.run()
        else:
            rospy.logerr("No valid CSV path found in YAML file.")

    except rospy.ROSInterruptException:
        pass
