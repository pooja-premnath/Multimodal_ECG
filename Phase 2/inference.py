import os
import argparse
import json
import numpy as np
import torch

from utils.util import find_max_epoch, print_size, sampling_label, calc_diffusion_hyperparams
from models.SSSD_ECG import SSSD_ECG


def generate_four_leads(tensor):
    print("[INFO] Generating four leads from ECG data...")
    leadI = tensor[:, 0, :].unsqueeze(1)
    leadschest = tensor[:, 1:7, :]
    leadavf = tensor[:, 7, :].unsqueeze(1)

    leadII = (0.5 * leadI) + leadavf
    leadIII = -(0.5 * leadI) + leadavf
    leadavr = -(0.75 * leadI) - (0.5 * leadavf)
    leadavl = (0.75 * leadI) - (0.5 * leadavf)

    leads12 = torch.cat([leadI, leadII, leadschest, leadIII, leadavr, leadavl, leadavf], dim=1)
    print("[INFO] Four leads generated successfully.")
    return leads12


def generate(output_directory, num_samples, ckpt_path, ckpt_iter):
    print("[INFO] Starting the generation process...")

    # Generate experiment (local) path
    local_path = "ch{}_T{}_betaT{}".format(model_config["res_channels"], 
                                           diffusion_config["T"], 
                                           diffusion_config["beta_T"])
    print(f"[INFO] Generated local path: {local_path}")

    # Prepare output directory
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print(f"[INFO] Output directory created: {output_directory}")

    # Map diffusion hyperparameters to GPU
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()
    print("[INFO] Diffusion hyperparameters mapped to GPU.")

    # Initialize model
    net = SSSD_ECG(**model_config).cuda()
    print_size(net)
    print("[INFO] Model initialized successfully.")

    # Load checkpoint
    ckpt_path = os.path.join(ckpt_path, local_path)
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_path)
    model_path = os.path.join(ckpt_path, '{}.pkl'.format(ckpt_iter))
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] Successfully loaded model at iteration {ckpt_iter}")
    except:
        raise Exception(f"[ERROR] No valid model found at {model_path}")

    # Load labels
    print("[INFO] Loading test labels...")
    labels = np.load('test_labels.npy')
    label_splits = [labels[i:i + 400] for i in range(0, len(labels), 400)]
    
    print("[INFO] Label file loaded successfully.")

    # Generate samples for each label batch
    for i, label in enumerate(label_splits):
        print(f"[INFO] Generating samples for batch {i + 1}...")
        
        cond = torch.from_numpy(label).cuda().float()

        # Start inference
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        generated_audio = sampling_label(net, (num_samples, 8, 1000), 
                                         diffusion_hyperparams,
                                         cond=cond)

        generated_audio12 = generate_four_leads(generated_audio)

        end.record()
        torch.cuda.synchronize()
        print(f"[INFO] Generated {num_samples} samples in {int(start.elapsed_time(end)/1000)} seconds.")

        # Save generated samples
        sample_filename = f"{i}_samples.npy"
        sample_out_path = os.path.join(ckpt_path, sample_filename)
        np.save(sample_out_path, generated_audio12.detach().cpu().numpy())
        print(f"[INFO] Saved generated samples: {sample_filename}")

        label_filename = f"{i}_labels.npy"
        label_out_path = os.path.join(ckpt_path, label_filename)
        np.save(label_out_path, cond.detach().cpu().numpy())
        print(f"[INFO] Saved label data: {label_filename}")


if __name__ == "__main__":
    print("[INFO] Parsing command-line arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/config_SSSD_ECG.json',
                        help='JSON file for configuration')
    parser.add_argument('-ckpt_iter', '--ckpt_iter', default=100000,
                        help='Which checkpoint to use; assign a number or "max"')
    parser.add_argument('-n', '--num_samples', type=int, default=400,
                        help='Number of utterances to be generated')
    args = parser.parse_args()

    print(f"[INFO] Loading configuration from {args.config}...")
    with open(args.config) as f:
        config = json.load(f)
    print("[INFO] Configuration loaded successfully.")

    # Extract configurations
    gen_config = config['gen_config']
    train_config = config["train_config"]  # Training parameters
    trainset_config = config["trainset_config"]  # To load trainset
    diffusion_config = config["diffusion_config"]  # Basic hyperparameters
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)  # All diffusion hyperparameters
    model_config = config['wavenet_config']

    print("[INFO] Configuration parameters initialized.")

    # Call generate function
    print("[INFO] Starting the ECG generation process...")
    generate(gen_config["output_directory"],
             args.num_samples,
             gen_config["ckpt_path"],
             args.ckpt_iter)
    print("[INFO] ECG generation process completed successfully.")
