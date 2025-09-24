import runpod
import torch
import os
import imageio
from easydict import EasyDict

# Import your WAN modules
from wan.text2video import WanT2V
from wan.speech2video import WanS2V
from wan.image2video import WanI2V
from wan.textimage2video import WanTI2V

# Load default configs (replace with actual configs from your repo)
from wan.configs import default_config as config  

# Initialize models once (so they don’t reload every request)
t2v_model = WanT2V(EasyDict(config), checkpoint_dir="checkpoints", device_id=0)
s2v_model = WanS2V(EasyDict(config), checkpoint_dir="checkpoints", device_id=0)
i2v_model = WanI2V(EasyDict(config), checkpoint_dir="checkpoints", device_id=0)
ti2v_model = WanTI2V(EasyDict(config), checkpoint_dir="checkpoints", device_id=0)

def save_video(tensor, out_path="output.mp4", fps=24):
    """Save generated tensor as video file."""
    frames = tensor.permute(1, 2, 3, 0).cpu().numpy()  # (frames, H, W, 3)
    imageio.mimwrite(out_path, frames, fps=fps, quality=8)
    return out_path

def handler(job):
    """Main RunPod handler."""
    inp = job["input"]
    mode = inp.get("mode", "text2video")  # Choose generation mode

    width = inp.get("width", 1280)
    height = inp.get("height", 720)
    frames = inp.get("frames", 81)

    if mode == "text2video":
        prompt = inp.get("prompt", "a cinematic battle scene")
        video_tensor = t2v_model.generate(
            input_prompt=prompt,
            size=(width, height),
            frame_num=frames
        )

    elif mode == "speech2video":
        audio_file = inp.get("audio_file")
        video_tensor = s2v_model.generate(audio_file, size=(width, height))

    elif mode == "image2video":
        image_file = inp.get("image_file")
        video_tensor = i2v_model.generate(image_file, size=(width, height))

    elif mode == "textimage2video":
        prompt = inp.get("prompt")
        image_file = inp.get("image_file")
        video_tensor = ti2v_model.generate(prompt, image_file, size=(width, height))

    else:
        return {"error": f"Unknown mode: {mode}"}

    # Save video file
    out_path = f"/tmp/{job['id']}.mp4"
    save_video(video_tensor, out_path)

    return {"video_path": out_path}

# Start serverless worker
runpod.serverless.start({"handler": handler})
