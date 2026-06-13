# play_isaac.py
# Standalone Python script for Windows Isaac Sim / Isaac Lab
# To run this script, execute from CMD / PowerShell:
#     isaaclab.bat -p play_isaac.py
# Or:
#     C:\Users\<username>\AppData\Local\ov\pkg\isaac-sim-4.X.X\python.bat play_isaac.py

import os
import sys
import yaml
import numpy as np
import onnxruntime as ort

# ====================================================================
# 1. Start Isaac Sim SimulationApp first
# ====================================================================
print("[Isaac Sim] Starting simulation application...")
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})  # Set False to open GUI

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.prims import RigidPrim
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.utils.types import ArticulationAction
import carb

# Get current folder path
DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))

# ====================================================================
# 2. Load Deployment Configurations
# ====================================================================
config_path = os.path.join(DEPLOY_DIR, "deploy_config.yaml")
if not os.path.exists(config_path):
    print(f"[Error] Configuration file not found at: {config_path}")
    simulation_app.close()
    sys.exit(1)

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

action_scale = float(cfg["action_scale"])
clip_obs = float(cfg["clip_obs"])
default_angles = np.deg2rad(np.asarray(cfg["default_angles_deg"], dtype=np.float32))
joint_names = cfg["joint_names"]
num_joints = len(joint_names)

print(f"[Config] Loaded {num_joints} joints configurations. action_scale={action_scale:.4f}")

# ====================================================================
# 3. Setup Simulation Scene
# ====================================================================
print("[Isaac Sim] Constructing scene...")
# Setup World with matching physics timestep (240 Hz) and control frequency (20 Hz)
world = World(
    stage_units_in_meters=1.0,
    physics_dt=1.0 / 240.0,
    rendering_dt=1.0 / 20.0,
)
world.scene.add_default_ground_plane()

# Robot USD Model path
hand_usd_path = r"D:\isaacsim_ws\right_sharpa_wave.usd"
if not os.path.exists(hand_usd_path):
    print(f"[Warning] Hand USD not found at: {hand_usd_path}")

# Add Articulation Hand (rotation/translation omitted to use defaults from USD)
prim_utils.create_prim(
    prim_path="/World/SharpaHand",
    usd_path=hand_usd_path,
)

# Dynamically locate the hand articulation prim path in the loaded USD stage
hand_prim_path = "/World/SharpaHand/right_hand_C_MC/right_hand_C_MC"
for prim in world.stage.Traverse():
    path = str(prim.GetPath())
    if path.endswith("right_hand_C_MC/right_hand_C_MC"):
        hand_prim_path = path
        break

print(f"[Physics] Detected hand Articulation path: {hand_prim_path}")

# Compute the default world pose of the hand base from USD stage before reset
from pxr import UsdGeom, UsdPhysics, Gf
hand_prim = world.stage.GetPrimAtPath(hand_prim_path)
xformable = UsdGeom.Xformable(hand_prim)
world_matrix = xformable.ComputeLocalToWorldTransform(0)
position = world_matrix.ExtractTranslation()
rot = world_matrix.ExtractRotation()
quat = rot.GetQuaternion()

# Configure any FixedJoint to anchor the hand root at this computed default pose
for prim in world.stage.Traverse():
    if prim.IsA(UsdPhysics.FixedJoint):
        fixed_joint = UsdPhysics.FixedJoint(prim)
        fixed_joint.GetBody0Rel().SetTargets([])
        fixed_joint.GetLocalPos0Attr().Set(Gf.Vec3f(position))
        fixed_joint.GetLocalRot0Attr().Set(Gf.Quatf(quat.GetReal(), Gf.Vec3f(quat.GetImaginary())))
        fixed_joint.GetLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        fixed_joint.GetLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
        print(f"[Physics] Anchored FixedJoint {prim.GetPath()} to computed pose: pos={position}, rot={quat}")
hand = Articulation(
    prim_path=hand_prim_path,
    name="sharpa_hand",
)
world.scene.add(hand)

# Dynamically locate the object/cube prim path in the USD stage
object_prim_path = "/World/SharpaHand/scene/object/object"
for prim in world.stage.Traverse():
    path = str(prim.GetPath())
    if path.endswith("scene/object/object"):
        object_prim_path = path
        break

print(f"[Physics] Detected object prim path: {object_prim_path}")
object_cube = RigidPrim(
    prim_path=object_prim_path,
    name="object_cube",
)
world.scene.add(object_cube)

# Reset world to instantiate physics
world.reset()
hand.initialize()

# Configure joint drives: read default USD stiffness, and set damping to stiffness * 0.05
dof_props = hand.dof_properties
kps = np.asarray(dof_props["stiffness"], dtype=np.float32).reshape(1, -1)
kds = kps * 0.05
hand._articulation_view.set_gains(kps=kps, kds=kds)
print(f"[Physics] Configured joint drives using USD default stiffness and scaled damping (stiffness*0.05). Stiffness range: [{kps.min():.2f}, {kps.max():.2f}]")

# Map joint order between Isaac Sim (alphabetical) and Policy config
isaac_joint_names = hand.dof_names
print(f"[Joint Map] Isaac Sim joints: {isaac_joint_names}")
print(f"[Joint Map] Policy joints: {joint_names}")
policy_to_isaac_idx = [isaac_joint_names.index(name) for name in joint_names]

# Map default angles from policy order to Isaac Sim alphabetical joint order
default_angles_isaac = np.zeros(num_joints, dtype=np.float32)
for i, idx in enumerate(policy_to_isaac_idx):
    default_angles_isaac[idx] = default_angles[i]

# Set the initial / default pose of the hand joints
hand.set_joints_default_state(positions=default_angles_isaac)
hand.set_joint_positions(positions=default_angles_isaac)
print("[Physics] Set hand joints to default initialization pose.")

# Get Joint Limits from Articulation properties and map to policy order
dof_limits_lower_isaac = hand.dof_properties["lower"]
dof_limits_upper_isaac = hand.dof_properties["upper"]
dof_limits_lower = np.array([dof_limits_lower_isaac[i] for i in policy_to_isaac_idx], dtype=np.float32)
dof_limits_upper = np.array([dof_limits_upper_isaac[i] for i in policy_to_isaac_idx], dtype=np.float32)

# ====================================================================
# 4. Load ONNX Policy Session
# ====================================================================
onnx_path = os.path.join(DEPLOY_DIR, "policy.onnx")
if not os.path.exists(onnx_path):
    print(f"[Warning] ONNX policy file not found at: {onnx_path}")
    print("Please copy the exported 'policy.onnx' from macOS to this folder.")
    sess = None
else:
    print(f"[ONNX] Loading policy from: {onnx_path}")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    onnx_in_shape = sess.get_inputs()[0].shape
    onnx_in_dim = int(onnx_in_shape[-1])
    print(f"[ONNX] Policy expected input dimension: {onnx_in_dim}")

# ====================================================================
# 5. Control and Inference Loop
# ====================================================================
prev_targets = default_angles.copy()
# Target limits are scaled by 0.9 relative to raw limits (matching dof_limits_scale in training config)
target_limits_lower = dof_limits_lower * 0.9
target_limits_upper = dof_limits_upper * 0.9

# Read initial object position dynamically from USD at startup
object_init_pos, _ = object_cube.get_world_pose()
print(f"[Physics] Initial object position loaded dynamically: {object_init_pos}")

# History buffers for policy observations (e.g. obs_lag_steps=3)
# Policy Frame Dim = joint_pos_rel (22) + last_actions (22) = 44 (or 49 if tactile is enabled)
# We can dynamically determine history length based on ONNX dimensions:
if sess is not None:
    priv_info_dim = 9
    policy_obs_dim = onnx_in_dim - priv_info_dim
    
    # Check if tactile sensor is included in the policy frame
    # (44 joints/actions + 5 tactile forces = 49)
    if policy_obs_dim % 49 == 0:
        policy_frame_dim = 49
        enable_tactile = True
    else:
        policy_frame_dim = 44
        enable_tactile = False
        
    obs_lag_steps = policy_obs_dim // policy_frame_dim
    print(f"[ONNX] Detected: policy_frame_dim={policy_frame_dim}, obs_lag_steps={obs_lag_steps}, enable_tactile={enable_tactile}")
else:
    policy_frame_dim = 44
    obs_lag_steps = 3
    enable_tactile = False

# Queue to hold the history of policy frames
history_buffer = []

print("[Isaac Sim] Starting execution. Close the UI window to exit.")
while simulation_app.is_running():
    world.step(render=True)
    
    if world.is_playing():
        # A. Read current joint states and map to policy joint order
        current_joint_pos_isaac = hand.get_joint_positions()
        if current_joint_pos_isaac is None:
            continue
        current_joint_pos = np.array([current_joint_pos_isaac[i] for i in policy_to_isaac_idx], dtype=np.float32)
        
        object_pose = object_cube.get_world_pose()
        if object_pose is None or object_pose[0] is None:
            continue
        object_pos, _ = object_pose
        
        # B. Construct current policy frame
        # Normalize joint positions to [-1, 1] using raw joint limits
        dof_norm = (2.0 * current_joint_pos - dof_limits_upper - dof_limits_lower) / (dof_limits_upper - dof_limits_lower + 1.0e-8)
        
        if enable_tactile:
            # Mock tactile sensors if not configured (5 sensors with force = 0.0)
            tactile = np.zeros(5, dtype=np.float32)
            policy_frame = np.concatenate([dof_norm, prev_targets, tactile])
        else:
            policy_frame = np.concatenate([dof_norm, prev_targets])
            
        # C. Maintain history window
        history_buffer.append(policy_frame)
        if len(history_buffer) < obs_lag_steps:
            # Fill with initial frame during startup
            while len(history_buffer) < obs_lag_steps:
                history_buffer.insert(0, policy_frame)
        elif len(history_buffer) > obs_lag_steps:
            history_buffer.pop(0)
            
        # D. Flatten history buffer
        flat_history = np.concatenate(history_buffer)
        
        # E. Assemble privileged info (Friction, mass, COM, scale)
        object_pos_delta = (object_pos - object_init_pos).astype(np.float32)
        friction = np.array([1.0], dtype=np.float32)
        mass = np.array([0.05], dtype=np.float32)
        com = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        scale = np.array([1.0], dtype=np.float32)
        
        priv_info = np.concatenate([object_pos_delta, friction, mass, com, scale])
        
        # F. Build final observation vector
        obs = np.concatenate([flat_history, priv_info]).astype(np.float32)
        obs = np.clip(obs, -clip_obs, clip_obs)
        
        # G. Query ONNX Policy
        if sess is not None:
            action = sess.run(None, {inp_name: obs[None, :]})[0][0]
            action = action.astype(np.float32)
            
            # H. Calculate joint target position (in policy order)
            target_joint_pos = action * action_scale + prev_targets
            target_joint_pos = np.clip(target_joint_pos, target_limits_lower, target_limits_upper)
            
            # Update prev_targets for next step
            prev_targets = target_joint_pos.copy()
            
            # Map target positions back to Isaac Sim alphabetical joint order
            target_joint_pos_isaac = np.zeros(num_joints, dtype=np.float32)
            for i, idx in enumerate(policy_to_isaac_idx):
                target_joint_pos_isaac[idx] = target_joint_pos[i]
            
            # Set targets in Isaac Sim joint order
            hand.apply_action(ArticulationAction(joint_positions=target_joint_pos_isaac))

simulation_app.close()
