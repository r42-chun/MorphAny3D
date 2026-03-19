import os
import trimesh
import numpy as np
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils.morphing_utils import run_morphing

# 1. SETUP
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large").cuda()

def render_seed_image(obj_path, out_path):
    """Renders a 3D file to a white-background PNG for the AI to read."""
    mesh = trimesh.load(obj_path)
    # Simple trick: Use trimesh's scene to get a quick screenshot
    scene = mesh.scene()
    data = scene.save_image(resolution=(512, 512))
    with open(out_path, "wb") as f:
        f.write(data)
    return Image.open(out_path).convert("RGB")

# 2. YOUR FILES
source_obj = "./inputs/rbc.obj"
target_obj = "./inputs/tcell.obj"

print("--- Rendering Seed Images ---")
src_img = render_seed_image(source_obj, "./assets/src_seed.png")
tar_img = render_seed_image(target_obj, "./assets/tar_seed.png")

# 3. CONFIGURE MORPH (Optimized for your RTX A5000)
params = {
    "morphing_num": 30,         # More frames for smoother motion
    "init_morphing_flag": False,
    "ss_mca_flag": True,        # Keep structure clean
    "slat_mca_flag": True,      # Keep details clean
    "ss_tfsa_flag": True,       # Prevent jitter
    "slat_tfsa_flag": True,
    "oc_flag": True,            # Align their rotations automatically
    "save_cache_path": "./outputs/cache_temp",
    "rm_cache": False           # Keep the .pt files so we can save meshes!
}

# 4. RUN
print("--- Starting AI Morphing ---")
# This will call the run_morphing function we discussed earlier
# run_morphing(pipeline, src_img, tar_img, params, 0, "./outputs/my_morph", "CustomMorph")