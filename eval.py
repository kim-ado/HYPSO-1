
import os
import importlib
import pickle
import pandas as pd
import numpy as np
import scipy.io as sio


import metric as me

importlib.reload(me)


def normalize_image(image, min_val=None, max_val=None):
    if min_val is None:
        min_val = np.min(image)
    if max_val is None:
        max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)

def combine_patches_blured(parent_folder, key):
    patches = []
    for folder_name in os.listdir(parent_folder):
        image_folder = os.path.join(parent_folder, folder_name)
        for file in os.listdir(image_folder):
            file_path = os.path.join(image_folder, file)
            if file_path.endswith('.mat'):
                patch = sio.loadmat(file_path)[key]
                patch_rgb = patch[:, :, :]
                patches.append(patch_rgb)

    patches = np.array(patches).reshape((11, 3, *patch_rgb.shape))
    image = normalize_image(np.concatenate(np.concatenate(patches, axis=1), axis=1))

    return image

def combine_patches(image_paths, patch_key_prefix):
    patches = []

    data = sio.loadmat(image_paths)
    for i in range(1, 133):  # Assuming there are 33 patches
        patch_key = f"{patch_key_prefix}_{str(i).zfill(2)}_pred"
        if patch_key in data:
            patch = data[patch_key]
            patch_rgb = patch[:, :, :]  # Assuming these are the R, G, B channels
            patches.append(patch_rgb)

    patches = np.array(patches).reshape((11, 3, *patch_rgb.shape))
    image = normalize_image(np.concatenate(np.concatenate(patches, axis=1), axis=1))

    return image


def evaluate():

    sbi = 43

    ml_fused = {
        "hyperpnn": '/home/kimado/master/DIP-HyperKite/Experiments/HyperPNN/hico_dataset/N_modules_2/final_prediction.mat',
        "hyperkite": '/home/kimado/master/DIP-HyperKite/Experiments/kitenetwithsk/hico_dataset/N_modules_2/final_prediction.mat',
        "darn": '/home/kimado/master/DIP-HyperKite/Experiments/DHP_DARN/hico_dataset/N_modules_2/final_prediction.mat'
    }

    blurred_path = ['/home/kimado/master/DIP-HyperKite/datasets/hico/edge_image.nc',
                     '/home/kimado/master/DIP-HyperKite/datasets/hico/H2010026150016.L1B_ISS.nc',
                     '/home/kimado/master/DIP-HyperKite/datasets/hico/H2010154045937.L1B_ISS.nc']

    shalg_results = {}
    for name, path in ml_fused.items():
        for paths in blurred_path:
            filename = os.path.basename(paths)
            image = normalize_image(combine_patches(path, filename)).astype(np.float32)
            blurred = normalize_image(combine_patches_blured(paths, 'blurred')).astype(np.float32)
            d_r_scores = []
            d_lambda_scores = []
            qnr_scores = []

            qnr, lam, dsr = me.qnr(blurred, image, sbi)
            qnr_scores.append(qnr)
            d_r_scores.append(dsr)
            d_lambda_scores.append(lam)

            qnr_scores = pd.DataFrame(qnr_scores)
            d_r_scores = pd.DataFrame(d_r_scores)
            d_lambda_scores = pd.DataFrame(d_lambda_scores)

            mean_qnr = np.mean(qnr_scores)
            mean_d_r = np.mean(d_r_scores)
            mean_d_lambda = np.mean(d_lambda_scores)

            shalg_results[name] = {
                "mean_qnr": mean_qnr,
                "mean_d_r": mean_d_r,
                "mean_d_lambda": mean_d_lambda
            }


        with open('ml.pkl', 'wb') as f:
            pickle.dump(shalg_results, f)
            
evaluate()
    