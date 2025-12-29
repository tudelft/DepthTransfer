import os
import numpy as np
import re
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# Path to the zip file and where to extract it
extraction_path = '/home/yhserver/aerial_gym_ws/aerial_gym/aerial_gym/control_data/states_data'

control_data_path = '/home/yhserver/aerial_gym_ws/aerial_gym/aerial_gym/control_data/control_data'

action_data_path = '/home/yhserver/aerial_gym_ws/aerial_gym/aerial_gym/control_data/actions_data'
reference_data_path = '/home/yhserver/aerial_gym_ws/aerial_gym/aerial_gym/control_data/reference_data'

# Function to extract numbers from filename for sorting
def numerical_sort(file):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(file)
    parts[1::2] = map(int, parts[1::2])  # Convert numeric strings to integers
    return parts

# List and sort all text files in the extraction directory numerically
extracted_files = sorted([f for f in os.listdir(extraction_path) if f.endswith('.txt')], key=numerical_sort)

control_files = sorted([f for f in os.listdir(control_data_path) if f.endswith('.txt')], key=numerical_sort)

action_files = sorted([f for f in os.listdir(action_data_path) if f.endswith('.txt')], key=numerical_sort)

reference_files = sorted([f for f in os.listdir(reference_data_path) if f.endswith('.txt')], key=numerical_sort)

robot_num_per_map = 1
# Function to read and process each file
# def process_file(file_path):
#     try:
#         # Load data as float to maintain precision
#         data = np.loadtxt(file_path, dtype=float)
#         # Verify correct shape
#         if data.shape != (robot_num_per_map*10, 13):
#             print(f"Skipping {file_path}: Expected shape (80, 13), got {data.shape}.")
#             return None
#         # Reshape into timestamps for each robot
#         reshaped_data = data.reshape(10, robot_num_per_map, 13)
#         # Initialize a new array to hold the data with added Euler angles
#         extended_data = np.empty((reshaped_data.shape[0], reshaped_data.shape[1], 16))
#         extended_data[:, :, :13] = reshaped_data  # Copy original data
#         # Convert quaternion to Euler angles and append to the data
#         for i in range(reshaped_data.shape[0]):  # Iterate over timestamps
#             for j in range(reshaped_data.shape[1]):  # Iterate over robots
#                 quat = reshaped_data[i, j, 3:7]  # Extract quaternion
#                 euler = R.from_quat(quat).as_euler('xyz', degrees=True)
#                 extended_data[i, j, 13:16] = euler  # Append Euler angles
#         return [extended_data[:, i, :] for i in range(robot_num_per_map)]
#     except Exception as e:
#         print(f"Error processing {file_path}: {e}")
#         return None
    
def process_file(file_path):
    try:
        # Load data as float to maintain precision
        data = np.loadtxt(file_path, dtype=float)
        # Verify correct shape
        if data.shape != (robot_num_per_map*10, 12):
            print(f"Skipping {file_path}: Expected shape (80, 12), got {data.shape}.")
            return None
        # Reshape into timestamps for each robot
        reshaped_data = data.reshape(10, robot_num_per_map, 12)
        # Initialize a new array to hold the data with added Euler angles
        # extended_data = np.empty((reshaped_data.shape[0], reshaped_data.shape[1], 15))
        # extended_data[:, :, :12] = reshaped_data  # Copy original data
        # Convert quaternion to Euler angles and append to the data
        # for i in range(reshaped_data.shape[0]):  # Iterate over timestamps
        #     for j in range(reshaped_data.shape[1]):  # Iterate over robots
        #         quat = reshaped_data[i, j, 3:7]  # Extract quaternion
        #         euler = R.from_quat(quat).as_euler('xyz', degrees=True)
        #         extended_data[i, j, 13:16] = euler  # Append Euler angles
        return [reshaped_data[:, i, :] for i in range(robot_num_per_map)]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Initialize lists to hold combined data for each robot
robots_data = [np.array([]).reshape(0, 12) for _ in range(robot_num_per_map)]

# Process each sorted file and compile data for each robot
for filename in extracted_files:
    full_path = os.path.join(extraction_path, filename)
    robot_data = process_file(full_path)
    if robot_data is not None:
        for idx, data in enumerate(robot_data):
            robots_data[idx] = np.vstack((robots_data[idx], data))

# Print one of the robot arrays for verification, e.g., Robot 1
print("Data for Robot 1 after appending Euler angles:")
print(robots_data[0])
print(f"Shape: {robots_data[0].shape}")


def process_control(file_path):
    try:
        # Load data as float to maintain precision
        data = np.loadtxt(file_path, dtype=float)
        # Verify correct shape
        if data.shape != (robot_num_per_map*10, 7):
            print(f"Skipping {file_path}: Expected shape (80, 4), got {data.shape}.")
            return None
        # Reshape into timestamps for each robot
        reshaped_data = data.reshape(10, robot_num_per_map, 7)
        return [reshaped_data[:, i, :] for i in range(robot_num_per_map)]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

control_data = [np.array([]).reshape(0, 7) for _ in range(robot_num_per_map)]

for filename in control_files:
    full_path = os.path.join(control_data_path, filename)
    thrust_data = process_control(full_path)
    if thrust_data is not None:
        for idx, data in enumerate(thrust_data):
            control_data[idx] = np.vstack((control_data[idx], data))

# Print one of the robot arrays for verification, e.g., Robot 1
print("Data for Robot 1 after appending Euler angles:")
print(control_data[0])
print(f"Shape: {control_data[0].shape}")

def process_actions(file_path):
    try:
        # Load data as float to maintain precision
        data = np.loadtxt(file_path, dtype=float)
        # Verify correct shape
        if data.shape != (robot_num_per_map*10, 4):
            print(f"Skipping {file_path}: Expected shape (80, 4), got {data.shape}.")
            return None
        # Reshape into timestamps for each robot
        reshaped_data = data.reshape(10, robot_num_per_map, 4)
        return [reshaped_data[:, i, :] for i in range(robot_num_per_map)]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

actions_data = [np.array([]).reshape(0, 4) for _ in range(robot_num_per_map)]

for filename in action_files:
    full_path = os.path.join(action_data_path, filename)
    action_data = process_actions(full_path)
    if action_data is not None:
        for idx, data in enumerate(action_data):
            actions_data[idx] = np.vstack((actions_data[idx], data))

# Print one of the robot arrays for verification, e.g., Robot 1
print("Data for Robot 1 after appending Euler angles:")
print(actions_data[0])
print(f"Shape: {actions_data[0].shape}")
# save the data every 10 steps to a txt file
index = np.arange(0, actions_data[0].shape[0], 10)
np.savetxt('actions_data.txt', actions_data[0][index, :], fmt='%.6f')

def process_references(file_path):
    try:
        # Load data as float to maintain precision
        data = np.loadtxt(file_path, dtype=float)
        # Verify correct shape
        if data.shape != (robot_num_per_map*10, 15):
            print(f"Skipping {file_path}: Expected shape (80, 6), got {data.shape}.")
            return None
        # Reshape into timestamps for each robot
        reshaped_data = data.reshape(10, robot_num_per_map, 15)
        return [reshaped_data[:, i, :] for i in range(robot_num_per_map)]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    
references_data = [np.array([]).reshape(0, 15) for _ in range(robot_num_per_map)]

for filename in reference_files:
    full_path = os.path.join(reference_data_path, filename)
    reference_data = process_references(full_path)
    if reference_data is not None:
        for idx, data in enumerate(reference_data):
            references_data[idx] = np.vstack((references_data[idx], data))

# Print one of the robot arrays for verification, e.g., Robot 1
print("Data for Robot 1 after appending Euler angles:")
print(references_data[0])
print(f"Shape: {references_data[0].shape}")


# Select the first three columns which might represent x, y, z coordinates or similar
robot_id = 0

x = robots_data[robot_id][:, 0]
y = robots_data[robot_id][:, 1]
z = robots_data[robot_id][:, 2]

roll = robots_data[robot_id][:, 3] * 180 / np.pi
pitch = robots_data[robot_id][:, 4] * 180 / np.pi
yaw = robots_data[robot_id][:, 5] * 180 / np.pi

vx = robots_data[robot_id][:, 6]
vy = robots_data[robot_id][:, 7]
vz = robots_data[robot_id][:, 8]

# calculate linear acceleration according to the velocity
ax = np.gradient(vx) * 100
ay = np.gradient(vy) * 100
az = np.gradient(vz) * 100

# ang_vx = robots_data[robot_id][:, 10]
# ang_vy = robots_data[robot_id][:, 11]
# ang_vz = robots_data[robot_id][:, 12]

output_thrust = control_data[robot_id][:, 0]

torque_command = control_data[robot_id][:, 1:4]

# rot_err = control_data[robot_id][:, 4:7]

angvel_err = control_data[robot_id][:, 4:7]

rotation_matrices = R.from_euler('xyz', robots_data[robot_id][:, 3:6], degrees=True).as_matrix()
rotation_matrix_transpose = rotation_matrices.transpose((0, 2, 1))
accel_command = actions_data[robot_id][:, :3]
accel_command[:, 2] += 1
forces_command = accel_command

ang_vel_body = robots_data[robot_id][:, 9:12]
# ang_vel_body = np.einsum('ijk,ik->ij', rotation_matrix_transpose, ang_vel)

ax_body = np.einsum('ijk,ik->ij', rotation_matrix_transpose, np.array([ax, ay, az]).transpose())

c_phi_s_theta = forces_command[:, 0]
s_phi = -forces_command[:, 1]
c_phi_c_theta = forces_command[:, 2]

pitch_setpoint = np.arctan2(c_phi_s_theta, c_phi_c_theta)
roll_setpoint = np.arctan2(s_phi, np.sqrt(c_phi_c_theta**2 + c_phi_s_theta**2))
yaw_setpoint = robots_data[robot_id][:, 5]

euler_setpoints = np.zeros((forces_command.shape[0], 3))
euler_setpoints[:, 0] = references_data[robot_id][:, 0] * 180 / np.pi
euler_setpoints[:, 1] = references_data[robot_id][:, 1] * 180 / np.pi
euler_setpoints[:, 2] = references_data[robot_id][:, 2] * 180 / np.pi

rates_setpoints = np.zeros((forces_command.shape[0], 3))
rates_setpoints[:, 0] = references_data[robot_id][:, 3]
rates_setpoints[:, 1] = references_data[robot_id][:, 4]
rates_setpoints[:, 2] = references_data[robot_id][:, 5]
# euler_setpoints[:, 0] = roll_setpoint * 180 / np.pi
# euler_setpoints[:, 1] = pitch_setpoint * 180 / np.pi
# euler_setpoints[:, 2] = yaw_setpoint

rotation_matrices_desired = R.from_euler('xyz', euler_setpoints, degrees=True).as_matrix()
rotation_matrices_desired_transpose = rotation_matrices_desired.transpose((0, 2, 1))

rotmat_euler_to_body_rates = np.zeros_like(rotation_matrices)
s_pitch = np.sin(robots_data[robot_id][:, 4])
c_pitch = np.cos(robots_data[robot_id][:, 4])
s_roll = np.sin(robots_data[robot_id][:, 3])
c_roll = np.cos(robots_data[robot_id][:, 3])

rotmat_euler_to_body_rates[:, 0, 0] = 1.0
rotmat_euler_to_body_rates[:, 1, 1] = c_roll
rotmat_euler_to_body_rates[:, 0, 2] = -s_pitch
rotmat_euler_to_body_rates[:, 2, 1] = -s_roll
rotmat_euler_to_body_rates[:, 1, 2] = s_roll * c_pitch
rotmat_euler_to_body_rates[:, 2, 2] = c_roll * c_pitch

euler_angle_rates = np.zeros_like(robots_data[robot_id][:, 9:12])
euler_angle_rates[:, 2] = actions_data[robot_id][:, 3]
# print("rotmat_euler_to_body_rates: ", rotmat_euler_to_body_rates)
# print("euler_angle_rates: ", euler_angle_rates)
omega_desired_body = np.einsum('ijk,ik->ij', rotmat_euler_to_body_rates, euler_angle_rates)
# print("omega_desired_body: ", omega_desired_body)

desired_angvel_body_frame= np.einsum('ijk,ik->ij', rotation_matrix_transpose, np.einsum('ijk,ik->ij', rotation_matrices_desired, omega_desired_body))
# print("desired_angvel_body_frame: ", desired_angvel_body_frame.shape)

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(x, label='X coordinate')
plt.plot(y, label='Y coordinate')
plt.plot(z, label='Z coordinate')
plt.plot(references_data[robot_id][:, 6], linestyle=':', label='x_setpoint')
plt.plot(references_data[robot_id][:, 7], linestyle='-.', label='y_setpoint')
plt.plot(references_data[robot_id][:, 8], linestyle='--', label='z_setpoint')
plt.title('Plot of the First Three Columns for Robot 1')
plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.legend()

# Create a plot for the Euler angles
plt.figure(figsize=(10, 6))
plt.plot(roll, label='Roll (degrees)')
plt.plot(pitch, label='Pitch (degrees)')
plt.plot(yaw, label='Yaw (degrees)')
plt.plot(euler_setpoints[:, 0], linestyle=':', label='roll_setpoint')
plt.plot(euler_setpoints[:, 1], linestyle='-.', label='pitch_setpoint')
plt.plot(euler_setpoints[:, 2], linestyle='--', label='yaw_setpoint')
plt.plot(pitch_setpoint * 180 / np.pi, linestyle=':', label='raw_pitch_setpoint')
plt.plot(roll_setpoint * 180 / np.pi, linestyle='-.', label='raw_roll_setpoint')
# plt.plot(yaw_setpoint, linestyle='--', label='raw_yaw_setpoint')
plt.title('Plot of Euler Angles for Robot 1')
plt.xlabel('Timestamp')
plt.ylabel('Angle (degrees)')
plt.legend()

# Create a plot for the Euler angles
plt.figure(figsize=(10, 6))
plt.plot(vx, label='Vx (m/s)')
plt.plot(vy, label='Vy (m/s)')
plt.plot(vz, label='Vz (m/s)')
plt.plot(references_data[robot_id][:, 9], linestyle=':', label='vx_setpoint')
plt.plot(references_data[robot_id][:, 10], linestyle='-.', label='vy_setpoint')
plt.plot(references_data[robot_id][:, 11], linestyle='--', label='vz_setpoint')
plt.title('Plot of velocity for Robot 1')
plt.xlabel('Timestamp')
plt.ylabel('vel (m/s)')
plt.legend()

# Create a plot for the acceleration
plt.figure(figsize=(10, 6))
# plt.plot(ax_body[:, 0], label='Ax (m/s^2)')
# plt.plot(ay, label='Ay (m/s^2)')
# plt.plot(az, label='Az (m/s^2)')
# plt.plot(actions_data[0][:, 0] * 9.8, label='accel_x')
plt.plot(references_data[robot_id][:, 12], linestyle=':', label='ax_setpoint')
plt.plot(references_data[robot_id][:, 13], linestyle='-.', label='ay_setpoint')
plt.plot(references_data[robot_id][:, 14], linestyle='--', label='az_setpoint')
plt.plot(actions_data[0][:, 0] * 9.8, label='accel_x')
plt.plot(actions_data[0][:, 1] * 9.8, label='accel_y')
# plt.plot((actions_data[0][:, 2] - 1) * 9.81, label='accel_z')
# plt.plot((actions_data[0][:, 2] - 1) * 9.81, label='accel_z')
plt.title('Plot of acceleration for Robot 1')
plt.xlabel('Timestamp')
plt.ylabel('m/s^2')
plt.legend()

plt.figure(figsize=(10, 6))
plt.plot(ang_vel_body[:, 0], label='ang_vx')
plt.plot(ang_vel_body[:, 1], label='ang_vy')
plt.plot(ang_vel_body[:, 2], label='ang_vz')
plt.plot(rates_setpoints[:, 0], linestyle=':', label='rates_setpoints_x')
plt.plot(rates_setpoints[:, 1], linestyle='-.', label='rates_setpoints_y')
plt.plot(rates_setpoints[:, 2], linestyle='--', label='rates_setpoints_z')
plt.title('Plot of body rates for Robot 1')
plt.xlabel('Timestamp')
plt.ylabel('vel (m/s)')
plt.legend()

plt.figure(figsize=(10, 6))
plt.plot(output_thrust / 1.24, label='output_thrust')
plt.title('Plot of thrust for Robot 1')
plt.xlabel('Timestamp')
plt.ylabel('N')

plt.figure(figsize=(10, 6))
plt.plot(torque_command[:, 0], label='torque_x')
plt.plot(torque_command[:, 1], label='torque_y')
plt.plot(torque_command[:, 2], label='torque_z')
plt.title('Plot of torque command for Robot 1')
plt.xlabel('Timestamp')
plt.ylabel('N.m')

# plt.figure(figsize=(10, 6))
# plt.plot(rot_err[:, 0], label='rot_err_x')
# plt.plot(rot_err[:, 1], label='rot_err_y')
# plt.plot(rot_err[:, 2], label='rot_err_z')
# plt.title('Plot of rotation error for Robot 1')
# plt.xlabel('Timestamp')
# plt.ylabel('rad')

plt.figure(figsize=(10, 6))
plt.plot(angvel_err[:, 0], label='angvel_err_x')
plt.plot(angvel_err[:, 1], label='angvel_err_y')
plt.plot(angvel_err[:, 2], label='angvel_err_z')
plt.title('Plot of angular velocity error for Robot 1')
plt.xlabel('Timestamp')
plt.ylabel('rad/s')

# plot action data
plt.figure(figsize=(10, 6))
plt.plot(actions_data[0][:, 0], label='accel_x')
plt.plot(actions_data[0][:, 1], label='accel_y')
plt.plot(actions_data[0][:, 2], label='accel_z')
# plt.plot(actions_data[0][:, 3], label='yaw_rate')
# plot acceleration states
plt.title('Plot of action data for Robot 1')
plt.xlabel('Timestamp')
plt.ylabel('m/s^2 or rad/s')

# plt.figure(figsize=(10, 6))
# plt.plot(euler_setpoints[:, 0], label='roll_setpoint')
# plt.plot(euler_setpoints[:, 1], label='pitch_setpoint')
# plt.plot(euler_setpoints[:, 2], label='yaw_setpoint')
# plt.title('Plot of euler setpoints for Robot 1')
# plt.xlabel('Timestamp')
# plt.ylabel('rad')

plt.show()