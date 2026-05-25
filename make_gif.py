"""Convert results/rollout_<traj>.mp4 → results/rollout_<traj>.gif for README."""
import os
import imageio.v3 as iio
import numpy as np

TRAJECTORIES = ["circle", "figure8", "lissajous"]

for traj in TRAJECTORIES:
    mp4 = f"results/rollout_{traj}.mp4"
    gif = f"results/rollout_{traj}.gif"
    if not os.path.exists(mp4):
        print(f"  [skip] {mp4} not found")
        continue

    frames = iio.imread(mp4, plugin="pyav")
    # Downsample every other frame to reduce file size
    frames = frames[::2]
    # Resize to 320 wide
    from PIL import Image
    resized = []
    for f in frames:
        img = Image.fromarray(f).resize((320, 240), Image.LANCZOS)
        resized.append(np.array(img))

    iio.imwrite(gif, resized, duration=80, loop=0)   # ~12.5 fps
    print(f"  {gif}  ({len(resized)} frames)")

print("Done.")
