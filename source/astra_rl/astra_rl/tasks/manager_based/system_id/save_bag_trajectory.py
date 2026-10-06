#!/usr/bin/env python3
import argparse
import math
import numpy as np

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t0, t1), math.asin(t2), math.atan2(t3, t4)

def parse_mcap_bag(bag_path, output_file):
    print(f"Opening bag: {bag_path}")
    
    storage_options = StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    
    target_topics = [
        "/blueboat/control/thrust_command",
        "/localization/pose",
        "/localization/twist"
    ]
    reader.set_filter(StorageFilter(topics=target_topics))
    
    type_map = {topic_metadata.name: topic_metadata.type 
                for topic_metadata in reader.get_all_topics_and_types()}
    
    latest_port_thrust = 0.0
    latest_stbd_thrust = 0.0
    received_first_thrust = False
    
    pending_poses = {}
    pending_twists = {}
    
    trajectory_history = []
    thrust_history = []
    time_history = []

    def stamp_to_ns(stamp):
        return stamp.sec * 10**9 + stamp.nanosec

    def check_match(stamp_ns):
        if stamp_ns in pending_poses and stamp_ns in pending_twists:
            pose_msg = pending_poses.pop(stamp_ns)
            twist_msg = pending_twists.pop(stamp_ns)
            
            if not received_first_thrust:
                return

            t_sec = pose_msg.header.stamp.sec
            t_nanosec = pose_msg.header.stamp.nanosec
            timestamp = t_sec + (t_nanosec * 1e-9)

            x = pose_msg.pose.position.x
            y = pose_msg.pose.position.y
            _, _, yaw = euler_from_quaternion(
                pose_msg.pose.orientation.x,
                pose_msg.pose.orientation.y,
                pose_msg.pose.orientation.z,
                pose_msg.pose.orientation.w
            )

            u = twist_msg.twist.linear.x
            v = twist_msg.twist.linear.y
            r = twist_msg.twist.angular.z

            trajectory_history.append([x, y, yaw, u, v, r])
            thrust_history.append([latest_port_thrust, latest_stbd_thrust])
            time_history.append(timestamp)

    while reader.has_next():
        topic, data, _ = reader.read_next()
        
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)

        if topic == "/blueboat/control/thrust_command":
            latest_port_thrust = msg.port_thrust_newtons
            latest_stbd_thrust = msg.starboard_thrust_newtons
            received_first_thrust = True

        elif topic == "/localization/pose":
            stamp_ns = stamp_to_ns(msg.header.stamp)
            pending_poses[stamp_ns] = msg
            if len(pending_poses) > 30:
                pending_poses.pop(next(iter(pending_poses)))
            check_match(stamp_ns)

        elif topic == "/localization/twist":
            stamp_ns = stamp_to_ns(msg.header.stamp)
            pending_twists[stamp_ns] = msg
            if len(pending_twists) > 30:
                pending_twists.pop(next(iter(pending_twists)))
            check_match(stamp_ns)

    np.savez(
        output_file, 
        time=np.array(time_history, dtype=np.float64),
        thrust=np.array(thrust_history, dtype=np.float32), 
        trajectory=np.array(trajectory_history, dtype=np.float32)
    )
    print(f"Saved {len(trajectory_history)} exact-matched steps to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse MCAP bag to extract system ID trajectories.")
    parser.add_argument("bag_path", help="Path to the .mcap file or folder.")
    parser.add_argument("--output", default="sysid_dataset.npz", help="Output .npz file name.")
    args = parser.parse_args()

    parse_mcap_bag(args.bag_path, args.output)
