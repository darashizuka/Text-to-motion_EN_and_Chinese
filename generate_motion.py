"""
Generate motions from text prompts using the trained mT5 + T2M-GPT model.

This script:
1. Takes text prompts (English or Chinese - mT5 is multilingual!)
2. Generates motion sequences using your trained model
3. Saves them as .npy files
4. Creates simple stick figure visualizations as .gif files

Usage (from T2M-GPT directory):
  python generate_motion.py --dataname t2m --resume-pth ./pretrained/VQVAE/net_best_fid.pth \
      --down-t 2 --depth 3 --dilation-growth-rate 3 --vq-act relu --block-size 51
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import options.option_transformer as option_trans
import models.vqvae as vqvae
import models.t2m_trans as trans
from mt5_encoder import MT5TextEncoder

args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

# Output directory for generated motions
output_dir = './generated_motions'
os.makedirs(output_dir, exist_ok=True)

##### ---- Load models ---- #####
print("Loading models...")

# mT5 encoder
mt5_encoder = MT5TextEncoder(
    model_name="google/mt5-small",
    output_dim=args.clip_dim,
    freeze_mt5=True
)

# VQ-VAE
net = vqvae.HumanVQVAE(
    args, args.nb_code, args.code_dim, args.output_emb_width,
    args.down_t, args.stride_t, args.width, args.depth,
    args.dilation_growth_rate
)
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

# GPT decoder
trans_encoder = trans.Text2Motion_Transformer(
    num_vq=args.nb_code, embed_dim=args.embed_dim_gpt,
    clip_dim=args.clip_dim, block_size=args.block_size,
    num_layers=args.num_layers, n_head=args.n_head_gpt,
    drop_out_rate=args.drop_out_rate, fc_rate=args.ff_rate
)

# Load checkpoint
ckpt_path = './output_GPT_Final/mt5_t2m/net_last.pth'
print(f'Loading checkpoint from {ckpt_path}')
ckpt = torch.load(ckpt_path, map_location='cpu')
trans_encoder.load_state_dict(ckpt['trans'], strict=True)
if 'mt5_encoder' in ckpt:
    mt5_encoder.load_state_dict(ckpt['mt5_encoder'], strict=True)
    print('Loaded mT5 encoder weights')

trans_encoder.eval()
trans_encoder.cuda()
mt5_encoder.eval()
mt5_encoder.cuda()

# Load mean/std for denormalization
datapath = './dataset/HumanML3D' if args.dataname == 't2m' else './dataset/KIT-ML'
mean = np.load(os.path.join(datapath, 'Mean.npy'))
std = np.load(os.path.join(datapath, 'Std.npy'))

##### ---- Text prompts ---- #####
# You can change these to anything you want!
prompts = [
    # # Simple English prompts
    "a person who is standing with his hands by his sides hops to his right and regains his balance on both feet.",
    "the man throws a big left handed punch in the air",
    
    "一个人双手垂于体侧站立，向右单脚跳跃，随后双脚着地并恢复平衡",
    "那个男人向空中挥出一记重重的左拳。"
    


    # "a person walks forward",
    # "a person jumps up",
    # "a person sits down on a chair",
    # "a person waves with their right hand",
    # "a person kicks with the left leg",

    # # Complex English prompts
    # "a person walks forward then turns around and walks back",
    # "a person bends down to pick something up from the ground",

    # # Chinese prompts (testing multilingual capability of mT5)
    # "一个人向前走",           # a person walks forward
    # "一个人跳起来",           # a person jumps up
    # "一个人坐下来",           # a person sits down

]


##### ---- HumanML3D skeleton structure ---- #####
# 22 joints for HumanML3D
t2m_kinematic_chain = [
    [0, 2, 5, 8, 11],       # right leg
    [0, 1, 4, 7, 10],       # left leg
    [0, 3, 6, 9, 12, 15],   # spine + head
    [9, 14, 17, 19, 21],    # right arm
    [9, 13, 16, 18, 20],    # left arm
]


def recover_joint_positions(motion_data, num_joints=22):
    """
    Extract joint positions from the 263-dim motion representation.
    The first num_joints*3 dimensions are joint positions (x,y,z).
    """
    # Denormalize
    motion_data = motion_data * std + mean

    # First 4 values are root velocity and rotation
    # Joint positions start after that, but the format is complex
    # For visualization, we use a simplified extraction
    # The motion representation has: root_rot_velocity(1), root_linear_velocity(2), root_y(1),
    # ric_data(21*3=63), rot_data(21*6=126), local_velocity(22*3=66), foot_contact(4)
    # ric_data gives us relative joint positions

    num_frames = motion_data.shape[0]
    joint_positions = np.zeros((num_frames, num_joints, 3))

    # Extract ric_data (relative joint positions, joints 1-21)
    # Starts at index 4, each joint has 3 values (x, y, z)
    ric_start = 4
    for j in range(num_joints - 1):
        idx = ric_start + j * 3
        joint_positions[:, j + 1, 0] = motion_data[:, idx]      # x
        joint_positions[:, j + 1, 1] = motion_data[:, idx + 1]  # y
        joint_positions[:, j + 1, 2] = motion_data[:, idx + 2]  # z

    # Root joint (joint 0) - use root_y for height
    joint_positions[:, 0, 1] = motion_data[:, 3]  # y (height)

    return joint_positions


def create_animation(joint_positions, title, save_path):
    """Create a stick figure animation and save as gif."""
    num_frames = joint_positions.shape[0]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()
        joints = joint_positions[frame]

        # Draw skeleton — swap Y and Z so the figure stands upright
        for chain in t2m_kinematic_chain:
            xs = joints[chain, 0]
            ys = joints[chain, 2]  # Z becomes horizontal
            zs = joints[chain, 1]  # Y becomes vertical (height)
            ax.plot(xs, ys, zs, 'b-', linewidth=2)

        # Draw joints
        ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1],
                   c='red', s=20)

        # Set axis limits
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([0, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y (height)')
        ax.set_title(f'{title}\nFrame {frame}/{num_frames}', fontsize=10)
        ax.view_init(elev=20, azim=120)

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50)
    ani.save(save_path, writer='pillow', fps=20)
    plt.close()
    print(f'  Saved animation: {save_path}')


##### ---- Generate motions ---- #####
print(f'\nGenerating motions for {len(prompts)} prompts...\n')

with torch.no_grad():
    for i, prompt in enumerate(prompts):
        print(f'[{i+1}/{len(prompts)}] "{prompt}"')

        # Encode text with mT5
        text_embedding = mt5_encoder([prompt])  # (1, 512)

        # Generate motion tokens with GPT
        motion_tokens = trans_encoder.sample(text_embedding, False)

        # Decode motion tokens with VQ-VAE
        motion = net.forward_decoder(motion_tokens)  # (1, num_frames, 263)
        motion = motion.cpu().numpy()[0]  # (num_frames, 263)

        # Save raw motion
        safe_name = prompt.replace(' ', '_')[:50]
        npy_path = os.path.join(output_dir, f'{i:02d}_{safe_name}.npy')
        np.save(npy_path, motion)
        print(f'  Saved motion: {npy_path} (shape: {motion.shape})')

        # Create visualization
        try:
            joints = recover_joint_positions(motion)
            gif_path = os.path.join(output_dir, f'{i:02d}_{safe_name}.gif')
            create_animation(joints, prompt, gif_path)
        except Exception as e:
            print(f'  Visualization failed: {e}')

print(f'\nDone! All outputs saved to {output_dir}/')
print('Files:')
for f in sorted(os.listdir(output_dir)):
    print(f'  {f}')
