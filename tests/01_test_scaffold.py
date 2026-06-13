import numpy as np
import torch
import mujoco
import gymnasium as gym

def test_scaffold():
    print("-" * 50)
    print(f"Numpy version: {np.__version__}")
    print(f"Torch version: {torch.__version__}")
    print(f"Mujoco version: {mujoco.__version__}")
    print(f"Gymnasium version: {gym.__version__}")
    
    # Check default device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Default device detected: {device}")
    
    # Simple XML compilation test
    xml_str = """
    <mujoco>
        <worldbody>
            <light name="top" pos="0 0 2"/>
            <geom name="floor" type="plane" size="1 1 0.1"/>
        </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml_str)
    mujoco.mj_forward(model, mujoco.MjData(model))
    print(f"MjModel compiled successfully! Number of geoms: {model.ngeom}")
    print("-" * 50)

if __name__ == "__main__":
    test_scaffold()
