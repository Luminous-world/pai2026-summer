# -*- coding: utf-8 -*-

# File: crane_vlm_agent.py
# Description: Autonomous Stacking Experiment using Qwen2.5-VL-7B-Instruct with Interactive Button UI, Calibration Probing, Action Memory, 1-Min Timelapse Exporter, and Full Prompt Logging for CRANE+ V2.

import os
import subprocess
import sys
import time
import random
import math
import types
import warnings
import gc
import json
import re

# Suppress warnings
warnings.filterwarnings("ignore")

# Print script file name and version at start
script_name = os.path.basename(__file__) if "__file__" in globals() else "crane_vlm_agent.py"
global_version = "6.7.0"
print(f"Executing: {script_name} | Version: {global_version}")
print(f"[SECTION 1] Starting section 1 (Global Version: {global_version})")

# Safe GPU memory release supporting both CUDA and MPS environments
def clear_gpu_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
    except Exception:
        pass

# Inject dummy ament_index_python module for ROS2-free xacro resolution
if "ament_index_python" not in sys.modules:
    ament_mod = types.ModuleType("ament_index_python")
    pkg_mod = types.ModuleType("ament_index_python.packages")

    def get_package_share_directory(pkg_name):
        base_dir = os.path.abspath("crane_plus")
        pkg_dir = os.path.join(base_dir, pkg_name)
        if os.path.exists(pkg_dir):
            return pkg_dir
        return os.path.abspath(pkg_name)

    pkg_mod.get_package_share_directory = get_package_share_directory
    ament_mod.packages = pkg_mod
    sys.modules["ament_index_python"] = ament_mod
    sys.modules["ament_index_python.packages"] = pkg_mod

# Helper function to execute shell commands with clean logging
def run_cmd_with_progress(cmd, description, env=None):
    start_time = time.time()
    if env is None:
        env = os.environ.copy()
    cpu_count = os.cpu_count() or 2
    env["MAKEFLAGS"] = f"-j{cpu_count}"

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    with open("execution_log.txt", "a", encoding="utf-8") as f_log:
        f_log.write(f"=== Task: {description} ===\n")
        for line in iter(process.stdout.readline, ''):
            f_log.write(line)
    process.stdout.close()
    return_code = process.wait()
    elapsed = time.time() - start_time
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)

# Install dependencies if missing
try:
    import torch
    import pybullet as p
    import xacro
    import ipywidgets as widgets
    from IPython.display import display, HTML
    from PIL import Image, ImageDraw
    import imageio
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
except ImportError:
    run_cmd_with_progress(
        [sys.executable, "-m", "pip", "install", "-q", "--prefer-binary",
         "torch", "transformers>=4.49.0", "qwen-vl-utils", "accelerate", "pybullet", "xacro",
         "Pillow", "imageio", "imageio-ffmpeg", "ipywidgets", "matplotlib"],
        "Installing VLM and Robotics dependencies"
    )
    import torch
    import pybullet as p
    import xacro
    import ipywidgets as widgets
    from IPython.display import display, HTML
    from PIL import Image, ImageDraw
    import imageio
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

import pybullet_data
import numpy as np
import io

# Main execution inside try...finally for strict memory release
try:
    clear_gpu_memory()

    # Load Qwen2.5-VL-7B-Instruct Model
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    print(f"[PROGRESS 10.0%] Loading VLM Model: {model_id} onto GPU...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto"
    )
    vlm_processor = AutoProcessor.from_pretrained(model_id)
    print(f"[PROGRESS 30.0%] VLM Model loaded successfully on {device}.")

    # Clone CRANE+ V2 repository and convert xacro to URDF
    repo_url = "https://github.com/rt-net/crane_plus.git"
    repo_dir = "crane_plus"

    if not os.path.exists(repo_dir):
        run_cmd_with_progress(["git", "clone", "--progress", repo_url], "Cloning CRANE+ V2 repository")

    xacro_file = os.path.join(repo_dir, "crane_plus_description", "urdf", "crane_plus.urdf.xacro")
    urdf_file = os.path.join(repo_dir, "crane_plus_description", "urdf", "crane_plus.urdf")

    if os.path.exists(xacro_file):
        doc = xacro.process_file(xacro_file)
        urdf_content = doc.toprettyxml(indent="  ").replace("package://crane_plus_description/", "")
        with open(urdf_file, "w") as f:
            f.write(urdf_content)

    # Initialize PyBullet Direct physics server
    if p.isConnected():
        p.disconnect()
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Create 80cm x 80cm laboratory worktable
    table_half_extents = [0.40, 0.40, 0.01]
    table_col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half_extents)
    table_vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=table_half_extents, rgbaColor=[1.0, 1.0, 1.0, 1.0])

    table_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=table_col_shape,
        baseVisualShapeIndex=table_vis_shape,
        basePosition=[0.20, 0.0, -0.01]
    )

    p.changeDynamics(
        table_id, -1,
        lateralFriction=3.0, spinningFriction=1.0, rollingFriction=1.0,
        restitution=0.0, contactStiffness=1e4, contactDamping=1e3
    )

    # Generate grid checkerboard texture
    img_size = 256
    grid_count = 16
    grid_tex_array = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    cell_size = img_size // grid_count

    for r in range(grid_count):
        for c in range(grid_count):
            if (r + c) % 2 == 0:
                grid_tex_array[r * cell_size:(r + 1) * cell_size, c * cell_size:(c + 1) * cell_size] = [235, 230, 220]
            else:
                grid_tex_array[r * cell_size:(r + 1) * cell_size, c * cell_size:(c + 1) * cell_size] = [215, 210, 200]

    grid_tex_file = "worktable_grid_texture.png"
    Image.fromarray(grid_tex_array).save(grid_tex_file)
    texture_id = p.loadTexture(grid_tex_file)
    p.changeVisualShape(table_id, -1, textureUniqueId=texture_id)

    # Load CRANE+ V2 Robot
    p.setAdditionalSearchPath(os.path.join(repo_dir, "crane_plus_description"))
    robot_base_pos = [0, 0, 0]
    robot_base_orient = p.getQuaternionFromEuler([0, 0, 0])
    robot_id = p.loadURDF(urdf_file, robot_base_pos, robot_base_orient, useFixedBase=True)

    # Parse URDF revolute joints dynamically
    num_joints = p.getNumJoints(robot_id)
    revolute_joint_indices = []
    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j)
        if info[2] == p.JOINT_REVOLUTE:
            revolute_joint_indices.append(j)

    base_joint_index = revolute_joint_indices[0] if len(revolute_joint_indices) > 0 else 2
    arm_joints = revolute_joint_indices[:4] if len(revolute_joint_indices) >= 4 else [2, 3, 4, 5]
    gripper_joints = revolute_joint_indices[4:] if len(revolute_joint_indices) > 4 else [6]
    end_effector_index = arm_joints[-1]

    for g_idx in gripper_joints:
        p.changeDynamics(
            robot_id, g_idx,
            lateralFriction=5.0, spinningFriction=2.0, rollingFriction=0.5, frictionAnchor=1
        )

    def create_block(size, position, color):
        half_extents = [s / 2.0 for s in size]
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
        block_id = p.createMultiBody(
            baseMass=0.20,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=position
        )
        p.changeDynamics(
            block_id, -1,
            lateralFriction=5.0, spinningFriction=2.0, rollingFriction=0.5,
            restitution=0.0, linearDamping=10.0, angularDamping=10.0, frictionAnchor=1
        )
        return block_id

    BASE_BLOCK_SIZE = 0.015
    block_center_r = 0.300
    block_red_scale = 2.8667  # 4.3cm block
    spawned_blocks = []

    def respawn_blocks():
        global spawned_blocks
        for b_id in spawned_blocks:
            p.removeBody(b_id)
        spawned_blocks.clear()

        red_color = [0.90, 0.10, 0.10, 1.0]
        red_size = BASE_BLOCK_SIZE * block_red_scale  # 4.3cm

        angle_rad = math.radians(30.0)
        pos_center = [block_center_r, 0.0, red_size / 2.0]
        pos_right = [block_center_r * math.cos(-angle_rad), block_center_r * math.sin(-angle_rad), red_size / 2.0]
        pos_left = [block_center_r * math.cos(angle_rad), block_center_r * math.sin(angle_rad), red_size / 2.0]

        for pos in [pos_center, pos_right, pos_left]:
            spawned_blocks.append(create_block([red_size, red_size, red_size], pos, red_color))

    respawn_blocks()

    # Camera Setup
    front_cam_target = [0.22, 0.0, 0.04]
    front_cam_distance = 0.45
    front_cam_yaw = 0
    front_cam_pitch = -30
    front_view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=front_cam_target, distance=front_cam_distance, yaw=front_cam_yaw, pitch=front_cam_pitch, roll=0, upAxisIndex=2
    )
    front_proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.05, farVal=3.0)
    wrist_proj_matrix = p.computeProjectionMatrixFOV(fov=75, aspect=1.0, nearVal=0.01, farVal=3.0)

    def get_wrist_camera_view_matrix():
        link_state = p.getLinkState(robot_id, end_effector_index)
        tcp_pos = link_state[0]
        cos_t = math.cos(cur_theta)
        sin_t = math.sin(cur_theta)
        cam_x = tcp_pos[0] - 0.01 * cos_t
        cam_y = tcp_pos[1] - 0.01 * sin_t
        cam_z = tcp_pos[2] + 0.095
        target_x = tcp_pos[0] + 0.06 * cos_t
        target_y = tcp_pos[1] + 0.06 * sin_t
        target_z = max(-0.02, tcp_pos[2] - 0.105)
        return p.computeViewMatrix(
            cameraEyePosition=[cam_x, cam_y, cam_z],
            cameraTargetPosition=[target_x, target_y, target_z],
            cameraUpVector=[0, 0, 1]
        )

    R_MIN, R_MAX = 0.14, 0.32
    THETA_MIN, THETA_MAX = -1.2, 1.2
    Z_MIN, Z_MAX = 0.001, 0.250

    STEP_R = 0.005
    STEP_THETA_RAD = math.radians(30.0)

    cur_r = 0.300
    cur_theta = 0.0
    cur_z = 0.120
    is_grasped = False
    current_gripper_torque = 3.0
    latest_action_log = "VLM Agent Initialized."
    latest_thought_log = "Initializing live UI and preparing calibration probe sequence..."

    captured_frames_buffer = []
    thought_process_text_logs = []
    recent_action_history = []  # Stores last 3 actions

    def get_actual_tcp_height():
        link_state = p.getLinkState(robot_id, end_effector_index)
        return link_state[0][2]

    def get_dynamic_max_height():
        current_block_size = BASE_BLOCK_SIZE * block_red_scale
        return min(Z_MAX, max(0.121, (3.0 * current_block_size) + 0.085))

    def check_landing_contact():
        ee_state = p.getLinkState(robot_id, end_effector_index)
        ee_pos = ee_state[0]
        ee_links = set(gripper_joints) | {end_effector_index}

        held_block_id = None
        if is_grasped:
            min_dist = 999.0
            for b_id in spawned_blocks:
                b_pos, _ = p.getBasePositionAndOrientation(b_id)
                dist = math.sqrt((b_pos[0] - ee_pos[0])**2 + (b_pos[1] - ee_pos[1])**2 + (b_pos[2] - ee_pos[2])**2)
                if dist < min_dist:
                    min_dist = dist
                    held_block_id = b_id
            if min_dist > 0.08:
                held_block_id = None

        contacts = p.getContactPoints()
        for c in contacts:
            bodyA, bodyB = c[1], c[2]
            linkA, linkB = c[3], c[4]
            is_ee_A = (bodyA == robot_id and linkA in ee_links)
            is_ee_B = (bodyB == robot_id and linkB in ee_links)
            is_held_A = (is_grasped and held_block_id is not None and bodyA == held_block_id)
            is_held_B = (is_grasped and held_block_id is not None and bodyB == held_block_id)

            if not (is_ee_A or is_ee_B or is_held_A or is_held_B):
                continue
            other_body = bodyB if (is_ee_A or is_held_A) else bodyA
            if other_body == table_id:
                return True, "Table Surface"
            if other_body in spawned_blocks and other_body != held_block_id:
                return True, "Stacked Block Surface"
        return False, ""

    def apply_ik_control(r, theta, z, grasp_closed):
        r = max(R_MIN, min(R_MAX, r))
        theta = max(THETA_MIN, min(THETA_MAX, theta))
        z = max(Z_MIN, min(Z_MAX, z))
        x = r * math.cos(theta)
        y = r * math.sin(theta)

        joint_poses = p.calculateInverseKinematics(
            robot_id, end_effector_index, targetPosition=[x, y, z],
            lowerLimits=[-1.2, -1.57, -1.57, -1.57], upperLimits=[1.2, 1.57, 1.57, 1.57],
            jointRanges=[2.4, 3.14, 3.14, 3.14], restPoses=[theta, -0.4, 1.2, 0.6],
            maxNumIterations=500, residualThreshold=1e-4
        )
        p.setJointMotorControl2(robot_id, base_joint_index, p.POSITION_CONTROL, targetPosition=theta, force=500, maxVelocity=5.0)
        for i in range(1, len(arm_joints)):
            p.setJointMotorControl2(robot_id, arm_joints[i], p.POSITION_CONTROL, targetPosition=joint_poses[i], force=500, maxVelocity=5.0)

        gripper_target = -0.50 if not grasp_closed else 0.35
        for g_idx in gripper_joints:
            p.setJointMotorControl2(robot_id, g_idx, p.POSITION_CONTROL, targetPosition=gripper_target, force=current_gripper_torque, maxVelocity=2.0)
        for _ in range(60):
            p.stepSimulation()

    def get_dual_camera_bytes():
        _, _, rgb_f, depth_f, _ = p.getCameraImage(300, 300, viewMatrix=front_view_matrix, projectionMatrix=front_proj_matrix, shadow=1, renderer=p.ER_TINY_RENDERER)
        rgb_arr_f = np.array(rgb_f, dtype=np.uint8).reshape((300, 300, 4))
        img_f = Image.fromarray(rgb_arr_f[:, :, :3], mode='RGB')
        buf_f = io.BytesIO()
        img_f.save(buf_f, format='JPEG', quality=85)

        wrist_vm = get_wrist_camera_view_matrix()
        _, _, rgb_w, depth_w, _ = p.getCameraImage(300, 300, viewMatrix=wrist_vm, projectionMatrix=wrist_proj_matrix, shadow=1, renderer=p.ER_TINY_RENDERER)
        rgb_arr_w = np.array(rgb_w, dtype=np.uint8).reshape((300, 300, 4))
        img_w = Image.fromarray(rgb_arr_w[:, :, :3], mode='RGB')
        buf_w = io.BytesIO()
        img_w.save(buf_w, format='JPEG', quality=85)
        return buf_f.getvalue(), buf_w.getvalue()

    def capture_composite_screen():
        _, _, rgb_f, _, _ = p.getCameraImage(300, 300, viewMatrix=front_view_matrix, projectionMatrix=front_proj_matrix, shadow=1, renderer=p.ER_TINY_RENDERER)
        arr_f = np.array(rgb_f, dtype=np.uint8).reshape((300, 300, 4))
        img_f = Image.fromarray(arr_f[:, :, :3], mode='RGB')

        wrist_vm = get_wrist_camera_view_matrix()
        _, _, rgb_w, _, _ = p.getCameraImage(300, 300, viewMatrix=wrist_vm, projectionMatrix=wrist_proj_matrix, shadow=1, renderer=p.ER_TINY_RENDERER)
        arr_w = np.array(rgb_w, dtype=np.uint8).reshape((300, 300, 4))
        img_w = Image.fromarray(arr_w[:, :, :3], mode='RGB')

        canvas = Image.new('RGB', (600, 620), color=(24, 25, 34))
        canvas.paste(img_f, (0, 20))
        canvas.paste(img_w, (300, 20))

        draw = ImageDraw.Draw(canvas)
        draw.text((10, 2), "[FRONT VIEW (Overhead)]", fill=(170, 170, 170))
        draw.text((310, 2), "[WRIST VIEW (1st Person)]", fill=(0, 229, 255))

        btn_y = 330
        draw.rectangle([10, btn_y, 290, btn_y + 25], fill=(30, 136, 229))
        draw.text((20, btn_y + 5), "Reach Forward (+0.5cm) [W]", fill=(255, 255, 255))
        draw.rectangle([305, btn_y, 585, btn_y + 25], fill=(30, 136, 229))
        draw.text((315, btn_y + 5), "Retract Arm (-0.5cm) [S]", fill=(255, 255, 255))

        btn_y += 30
        draw.rectangle([10, btn_y, 290, btn_y + 25], fill=(30, 136, 229))
        draw.text((20, btn_y + 5), "Rotate Left (+30deg) [A]", fill=(255, 255, 255))
        draw.rectangle([305, btn_y, 585, btn_y + 25], fill=(30, 136, 229))
        draw.text((315, btn_y + 5), "Rotate Right (-30deg) [D]", fill=(255, 255, 255))

        btn_y += 30
        draw.rectangle([10, btn_y, 290, btn_y + 25], fill=(0, 172, 193))
        draw.text((20, btn_y + 5), "Up to max height [I]", fill=(255, 255, 255))
        draw.rectangle([305, btn_y, 585, btn_y + 25], fill=(0, 172, 193))
        draw.text((315, btn_y + 5), "Down to ground [K]", fill=(255, 255, 255))

        btn_y += 30
        draw.rectangle([10, btn_y, 290, btn_y + 25], fill=(255, 179, 0))
        draw.text((20, btn_y + 5), "Grasp/Release [SPACE]", fill=(0, 0, 0))
        draw.rectangle([305, btn_y, 585, btn_y + 25], fill=(229, 57, 53))
        draw.text((315, btn_y + 5), "Reset Environment [R]", fill=(255, 255, 255))

        console_y = 480
        draw.rectangle([10, console_y, 585, 610], outline=(45, 48, 64), fill=(15, 16, 22))
        draw.text((20, console_y + 8), f"> Action: {latest_action_log}", fill=(0, 229, 255))

        thought_disp = latest_thought_log if len(latest_thought_log) < 70 else latest_thought_log[:67] + "..."
        draw.text((20, console_y + 26), f"AI Thought: {thought_disp}", fill=(255, 213, 79))

        actual_z = get_actual_tcp_height()
        gripper_disp = "[CLOSED]" if is_grasped else "[OPEN]"
        draw.text((20, console_y + 48),
                  f"ARM: r={cur_r:.3f}m | theta={math.degrees(cur_theta):.1f}deg | z={cur_z:.3f}m (actual={actual_z:.3f}m) | GRIPPER: {gripper_disp}",
                  fill=(170, 170, 170))

        return canvas

    # [ADDED START] v6.7.0: Restored interactive ipywidgets Control Buttons Grid with click event handlers
    bytes_front, bytes_wrist = get_dual_camera_bytes()
    widget_img_front = widgets.Image(value=bytes_front, format='jpeg', width=300, height=300)
    widget_img_wrist = widgets.Image(value=bytes_wrist, format='jpeg', width=300, height=300)
    status_html = widgets.HTML()

    # Create interactive control buttons
    btn_w = widgets.Button(description="Reach Forward (+0.5cm) [W]", button_style='primary', layout=widgets.Layout(width='295px'))
    btn_s = widgets.Button(description="Retract Arm (-0.5cm) [S]", button_style='primary', layout=widgets.Layout(width='295px'))
    btn_a = widgets.Button(description="Rotate Left (+30deg) [A]", button_style='primary', layout=widgets.Layout(width='295px'))
    btn_d = widgets.Button(description="Rotate Right (-30deg) [D]", button_style='primary', layout=widgets.Layout(width='295px'))
    btn_i = widgets.Button(description="Up to max height [I]", button_style='info', layout=widgets.Layout(width='295px'))
    btn_k = widgets.Button(description="Down to ground [K]", button_style='info', layout=widgets.Layout(width='295px'))
    btn_space = widgets.Button(description="Grasp/Release [SPACE]", button_style='warning', layout=widgets.Layout(width='295px'))
    btn_r = widgets.Button(description="Reset Environment [R]", button_style='danger', layout=widgets.Layout(width='295px'))

    def update_live_ui():
        apply_ik_control(cur_r, cur_theta, cur_z, is_grasped)
        bf, bw = get_dual_camera_bytes()
        widget_img_front.value = bf
        widget_img_wrist.value = bw

        actual_z = get_actual_tcp_height()
        gripper_disp = "[CLOSED]" if is_grasped else "[OPEN]"
        status_html.value = f"""
        <div style="background:#181922; border:1px solid #2d3040; border-radius:8px; padding:10px; width:600px; font-family:monospace; color:#E0E0E0;">
            <div style="color:#00E5FF; font-weight:bold;">> Action: {latest_action_log}</div>
            <div style="color:#FFD54F; margin-top:4px;">AI Thought: {latest_thought_log}</div>
            <div style="margin-top:6px; font-size:11px; color:#AAA;">
                ARM: r={cur_r:.3f}m | θ={math.degrees(cur_theta):.1f}° | z_cmd={cur_z:.3f}m (actual={actual_z:.3f}m) | GRIPPER: {gripper_disp}
            </div>
        </div>
        """

    # Action Execution Functions
    def action_reach(dr):
        global cur_r, latest_action_log
        cur_r = max(R_MIN, min(R_MAX, cur_r + dr))
        latest_action_log = f"Reach {'Forward' if dr>0 else 'Retract'} -> r={cur_r:.3f}m"
        apply_ik_control(cur_r, cur_theta, cur_z, is_grasped)
        update_live_ui()

    def action_rotate(d_theta_rad):
        global cur_theta, latest_action_log
        cur_theta = max(THETA_MIN, min(THETA_MAX, cur_theta + d_theta_rad))
        latest_action_log = f"Rotated Arm -> theta={math.degrees(cur_theta):.1f}deg"
        apply_ik_control(cur_r, cur_theta, cur_z, is_grasped)
        update_live_ui()

    def action_up_to_max_height():
        global cur_z, latest_action_log
        target_z = get_dynamic_max_height()
        latest_action_log = f"Lifting arm -> Up to max height (z={target_z:.3f}m)"
        start_z = cur_z
        steps = max(1, int(abs(target_z - start_z) / 0.005))
        for i in range(1, steps + 1):
            cur_z = start_z + (target_z - start_z) * (i / float(steps))
            apply_ik_control(cur_r, cur_theta, cur_z, is_grasped)
            if i % 3 == 0:
                frame = capture_composite_screen()
                captured_frames_buffer.append(np.array(frame))
                update_live_ui()
            time.sleep(0.01)

    def action_down_to_ground():
        global cur_z, latest_action_log
        target_z = 0.001
        latest_action_log = "Descending arm -> Down to ground"
        start_z = cur_z
        steps = max(1, int(abs(target_z - start_z) / 0.005))
        for i in range(1, steps + 1):
            cur_z = start_z + (target_z - start_z) * (i / float(steps))
            apply_ik_control(cur_r, cur_theta, cur_z, is_grasped)
            if is_grasped:
                contact, target_surf = check_landing_contact()
                if contact:
                    latest_action_log = f"Landing contact detected on [{target_surf}] at actual={get_actual_tcp_height():.3f}m."
                    break
            if i % 3 == 0:
                frame = capture_composite_screen()
                captured_frames_buffer.append(np.array(frame))
                update_live_ui()
            time.sleep(0.01)

    def action_grasp():
        global is_grasped, latest_action_log
        is_grasped = not is_grasped
        latest_action_log = f"Gripper -> {'CLOSED' if is_grasped else 'OPEN'}"
        apply_ik_control(cur_r, cur_theta, cur_z, is_grasped)
        update_live_ui()

    def action_reset_env():
        global cur_r, cur_theta, cur_z, is_grasped, latest_action_log
        cur_r, cur_theta, cur_z, is_grasped = 0.300, 0.0, 0.120, False
        apply_ik_control(cur_r, cur_theta, cur_z, is_grasped)
        respawn_blocks()
        latest_action_log = "Environment & Robot Pose Reset."
        update_live_ui()

    # Button Event Handlers
    def on_btn_w_clicked(b): action_reach(STEP_R)
    def on_btn_s_clicked(b): action_reach(-STEP_R)
    def on_btn_a_clicked(b): action_rotate(STEP_THETA_RAD)
    def on_btn_d_clicked(b): action_rotate(-STEP_THETA_RAD)
    def on_btn_i_clicked(b): action_up_to_max_height()
    def on_btn_k_clicked(b): action_down_to_ground()
    def on_btn_space_clicked(b): action_grasp()
    def on_btn_r_clicked(b): action_reset_env()

    btn_w.on_click(on_btn_w_clicked)
    btn_s.on_click(on_btn_s_clicked)
    btn_a.on_click(on_btn_a_clicked)
    btn_d.on_click(on_btn_d_clicked)
    btn_i.on_click(on_btn_i_clicked)
    btn_k.on_click(on_btn_k_clicked)
    btn_space.on_click(on_btn_space_clicked)
    btn_r.on_click(on_btn_r_clicked)

    # Combine camera views, interactive buttons grid, and status console
    fast_ui_layout = widgets.VBox([
        widgets.HBox([
            widgets.VBox([widgets.HTML("<b style='color:#AAA;'>[FRONT VIEW (Overhead)]</b>"), widget_img_front]),
            widgets.VBox([widgets.HTML("<b style='color:#00E5FF;'>[WRIST VIEW (1st Person)]</b>"), widget_img_wrist])
        ]),
        widgets.VBox([
            widgets.HBox([btn_w, btn_s]),
            widgets.HBox([btn_a, btn_d]),
            widgets.HBox([btn_i, btn_k]),
            widgets.HBox([btn_space, btn_r])
        ], layout=widgets.Layout(margin='10px 0px')),
        status_html
    ])
    display(fast_ui_layout)
    # [ADDED END] v6.7.0

    # System Prompts and Initial Conditions Setup
    system_prompt_text = (
        "=== EXPERIMENTAL SETUP & SYSTEM INSTRUCTIONS ===\n"
        "Robot Platform: CRANE+ V2 Manipulator (4 DOF Arm + Gripper)\n"
        "Task Goal: Stack 4.3cm red blocks as high as possible in 2-tier stacking within 2 minutes.\n"
        "Action Space & Rules:\n"
        "  - 'fwd': Reach arm forward (+0.5cm, r_max=0.320m)\n"
        "  - 'back': Retract arm backward (-0.5cm, r_min=0.140m)\n"
        "  - 'left': Rotate base arm left (+30 degrees)\n"
        "  - 'right': Rotate base arm right (-30 degrees)\n"
        "  - 'up': Lift arm to dynamic safe height (z ~ 0.214m)\n"
        "  - 'down': Descend arm down to ground/stacked surface (z ~ 0.001m with auto-stop when holding block)\n"
        "  - 'grasp': Toggle gripper between OPEN (release) and CLOSED (grasp block)\n"
        "  - 'reset': Reset environment and robot pose to default (r=0.300m, theta=0.0deg, z=0.120m)\n\n"
        "Constraint: Avoid repeating 'up' if already at max height. Pick up a block first before attempting to stack.\n"
        "Output Format Constraint: Respond strictly in JSON:\n"
        "{\n"
        "  \"thought_process\": \"<Analysis of camera view and current arm state>\",\n"
        "  \"action\": \"<one of: fwd, back, left, right, up, down, grasp, reset>\"\n"
        "}\n"
    )

    # Write Initial Instructions to Thought Log File
    thought_log_file = "vlm_thought_process_log.txt"
    with open(thought_log_file, "w", encoding="utf-8") as f:
        f.write("=== CRANE+ V2 VLM Autonomous Stacking - AI Thought Process Log ===\n\n")
        f.write(system_prompt_text + "\n")
        f.write("=" * 60 + "\n\n")

    # VLM Reasoning & Decision Routine
    def query_vlm_agent(screen_pil):
        history_str = ", ".join(recent_action_history[-3:]) if len(recent_action_history) > 0 else "None"
        user_prompt = (
            f"Recent Executed Actions: [{history_str}]\n"
            "Analyze the camera images (Front View overhead, Wrist View 1st person) and choose the next strategic action.\n"
            "Do NOT repeat 'up' if already at max height. If no block is held, lower the arm with 'down' to reach a block."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": screen_pil},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]

        text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = vlm_processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            generated_ids = vlm_model.generate(**inputs, max_new_tokens=150)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response_text = vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                return parsed.get("thought_process", "Reasoning..."), parsed.get("action", "down")
        except Exception:
            pass

        return "Executing default trial step based on state analysis.", "down"

    # Calibration Probe Phase (Warm-Up to teach action dynamics)
    print("[PROGRESS 35.0%] Running Calibration Probing Phase (Warm-up probe sequence)...")
    probe_actions = ["down", "up", "left", "reset"]
    for p_idx, p_act in enumerate(probe_actions):
        latest_thought_log = f"Calibration Probe {p_idx+1}/4: Testing action [{p_act}] for in-context grounding."
        if p_act == "down":
            action_down_to_ground()
        elif p_act == "up":
            action_up_to_max_height()
        elif p_act == "left":
            action_rotate(STEP_THETA_RAD)
        elif p_act == "reset":
            action_reset_env()

        frame = capture_composite_screen()
        captured_frames_buffer.append(np.array(frame))
        recent_action_history.append(f"probe_{p_act}")
        time.sleep(0.5)

    print("[PROGRESS 40.0%] Calibration Probing Complete. Starting Autonomous Stacking Loop...")

    # Main Autonomous Trial Loop
    max_experiment_steps = 15
    for step in range(1, max_experiment_steps + 1):
        progress_pct = 40.0 + (step / float(max_experiment_steps)) * 40.0

        frame_pil = capture_composite_screen()
        frame_np = np.array(frame_pil)
        captured_frames_buffer.append(frame_np)

        thought, selected_action = query_vlm_agent(frame_pil)
        latest_thought_log = thought
        recent_action_history.append(selected_action)

        # Log to file
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = (
            f"--- Autonomous Step {step}/{max_experiment_steps} [{timestamp_str}] ---\n"
            f"Selected Action: {selected_action}\n"
            f"Thought Process: {thought}\n"
            f"Arm Pose: r={cur_r:.3f}m, theta={math.degrees(cur_theta):.1f}deg, z={cur_z:.3f}m\n\n"
        )
        with open(thought_log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)

        # Execute chosen action
        if selected_action == "fwd":
            action_reach(STEP_R)
        elif selected_action == "back":
            action_reach(-STEP_R)
        elif selected_action == "left":
            action_rotate(STEP_THETA_RAD)
        elif selected_action == "right":
            action_rotate(-STEP_THETA_RAD)
        elif selected_action == "up":
            action_up_to_max_height()
        elif selected_action == "down":
            action_down_to_ground()
        elif selected_action == "grasp":
            action_grasp()
        elif selected_action == "reset":
            action_reset_env()
        else:
            action_down_to_ground()

        post_frame = capture_composite_screen()
        captured_frames_buffer.append(np.array(post_frame))
        update_live_ui()

    print("[PROGRESS 80.0%] Autonomous Trial Execution Completed.")

    # Process captured frames into an exact 1-minute (60-second, 20 FPS) timelapse MP4 video for 2x speed playback of a 2-minute trial duration
    timelapse_output_file = "vla_autonomous_stacking_timelapse_60s.mp4"
    target_duration_sec = 60.0
    target_fps = 20
    total_target_frames = int(target_duration_sec * target_fps)  # 1200 frames

    if len(captured_frames_buffer) > 0:
        print(f"[PROGRESS 81.0%] Processing {len(captured_frames_buffer)} frames into exact 1-minute timelapse video...")
        indices = np.linspace(0, len(captured_frames_buffer) - 1, total_target_frames).astype(int)

        tl_writer = imageio.get_writer(timelapse_output_file, fps=target_fps)
        start_time_tl = time.time()

        for idx_i, frame_idx in enumerate(indices):
            tl_writer.append_data(captured_frames_buffer[frame_idx])

            progress_val = 81.0 + ((idx_i + 1) / float(total_target_frames)) * 18.0
            if idx_i % max(1, int(total_target_frames / 1000)) == 0 or idx_i == total_target_frames - 1:
                elapsed = time.time() - start_time_tl
                rem_time = (elapsed / (idx_i + 1)) * (total_target_frames - (idx_i + 1)) if idx_i + 1 > 0 else 0
                sys.stdout.write(f"\r[PROGRESS {progress_val:.1f}%] Encoding Timelapse: {idx_i + 1}/{total_target_frames} frames (ETA: {rem_time:.1f}s)")
                sys.stdout.flush()

        tl_writer.close()
        print(f"\n[PROGRESS 99.0%] 1-minute timelapse video saved to '{timelapse_output_file}'.")

    print(f"[PROGRESS 100.0%] All tasks complete. Complete prompt & CoT log saved to '{thought_log_file}'.")

finally:
    clear_gpu_memory()
    if p.isConnected():
        p.disconnect()
    print("[LOG] GPU Memory and PyBullet Resources Successfully Cleaned Up.")