# inspect_usd.py
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World
import omni.isaac.core.utils.prims as prim_utils
import os

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
hand_usd_path = os.path.join(DEPLOY_DIR, "assets", "robots", "sharpa_wave", "right_sharpa_wave", "right_sharpa_wave.usd")

world = World(stage_units_in_meters=1.0)
prim_utils.create_prim(
    prim_path="/World/SharpaHand",
    usd_path=hand_usd_path,
)

stage = world.stage
print("--- Articulation Search ---")
for prim in stage.Traverse():
    # Check if this is an articulation root
    if prim.HasAPI("PhysxArticulationAPI") or prim.IsA("PhysicsArticulationRoot"):
        print(f"Found Articulation Root: {prim.GetPath()} ({prim.GetTypeName()})")

print("--- Prim Hierarchy under /World/SharpaHand ---")
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if path.startswith("/World/SharpaHand"):
        print(f"  {path} ({prim.GetTypeName()})")

simulation_app.close()
