import os
from inception_score import get_inception_score
from pytorch_fid import fid_score
from pytorch_lpips import lpips
import pytorch_lpips

from skimage.io import imread, imsave
from skimage.measure import compare_ssim

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

def l1_score(generated_images, reference_images):
    score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        score = np.abs(2 * (reference_image/255.0 - 0.5) - 2 * (generated_image/255.0 - 0.5)).mean()
        score_list.append(score)
    return np.mean(score_list)


def ssim_score(generated_images, reference_images):
    ssim_score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        ssim = compare_ssim(reference_image, generated_image, gaussian_weights=True, sigma=1.5,
                            use_sample_covariance=False, multichannel=True,
                            data_range=generated_image.max() - generated_image.min())
        ssim_score_list.append(ssim)
    return np.mean(ssim_score_list)


def lpips_score(generated_images, reference_images):

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

    ## Initializing the model
    loss_fn = lpips.LPIPS(net='alex', version='0.1')
    loss_fn.to(device)
    lpips_score_list = []
    for reference_image, generated_image, i in zip(reference_images, generated_images, tqdm(range(len(generated_images)))):
        # lpips = compare_ssim(reference_image, generated_image, gaussian_weights=True, sigma=1.5,
        #                     use_sample_covariance=False, multichannel=True,
        #                     data_range=generated_image.max() - generated_image.min())
        # Load images
        img0 = pytorch_lpips.im2tensor(reference_image)  # RGB image from [-1,1]
        img1 = pytorch_lpips.im2tensor(generated_image)

        img0 = img0.to(device)
        img1 = img1.to(device)

        # Compute distance
        with torch.no_grad():
            dist01 = loss_fn.forward(img0, img1)
            lpips_score_list.append(dist01.cpu().numpy())
    return np.mean(lpips_score_list)


def save_images(input_images, target_images, generated_images, names, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for images in zip(input_images, target_images, generated_images, names):
        res_name = str('_'.join(images[-1])) + '.png'
        imsave(os.path.join(output_folder, res_name), np.concatenate(images[:-1], axis=1))


def create_masked_image(names, images, annotation_file):
    import pose_utils
    masked_images = []
    df = pd.read_csv(annotation_file, sep=':')
    for name, image in zip(names, images):
        to = name[1]
        ano_to = df[df['name'] == to].iloc[0]

        kp_to = pose_utils.load_pose_cords_from_strings(ano_to['keypoints_y'], ano_to['keypoints_x'])

        mask = pose_utils.produce_ma_mask(kp_to, image.shape[:2])
        masked_images.append(image * mask[..., np.newaxis])

    return masked_images

def addBounding(image, bound=40):
    h, w, c = image.shape
    image_bound = np.ones((h, w+bound*2, c))*255
    image_bound = image_bound.astype(np.uint8)
    image_bound[:, bound:bound+w] = image

    return image_bound

def load_generated_images(images_folder):
    input_images = []
    target_images = []
    generated_images = []
    names = []
    for img_name in os.listdir(images_folder):
        img = imread(os.path.join(images_folder, img_name))
        w = int(img.shape[1] / 8) #h, w ,c

        input_images.append(addBounding(img[:, :w]))
        target_images.append(addBounding(img[:, 3*w:4*w]))
        generated_images.append(addBounding(img[:, 7*w:8*w]))

        assert img_name.endswith('_vis.png') or img_name.endswith('_vis.jpg'), 'unexpected img name: should end with _vis.png'
        img_name = img_name[:-8]
        img_name = img_name.split('___')
        assert len(img_name) == 2, 'unexpected img split: length 2 expect!'
        fr = img_name[0]
        to = img_name[1]

        names.append([fr, to])

    return input_images, target_images, generated_images, names


def test(generated_images_dir):
    f = open('score.txt', 'a+')
    # load images
    print ("Loading images...")

    input_images, target_images, generated_images, names = load_generated_images(generated_images_dir)

    print ("Compute inception score...")
    inception_score = get_inception_score(generated_images)
    print ("Inception score %s" % inception_score[0])

    print ("Compute structured similarity score (SSIM)...")
    structured_score = ssim_score(generated_images, target_images)
    print ("SSIM score %s" % structured_score)

    print("Compute FID score...")
    FID_score = fid_score.calculate_fid_given_paths([generated_images, target_images], 1, dims=2048)
    print("FID score %s" % FID_score)

    print("Compute LPIPS score...")
    LPIPS_score = lpips_score(generated_images, target_images)
    print("LPIPS score %s" % LPIPS_score)

    msg = "Inception score = %s; SSIM score = %s; FID score = %s; LPIPS score = %s" % (inception_score, structured_score, FID_score, LPIPS_score)
    print (msg)
    f.writelines('\nTarget image dir %s\n' % generated_images_dir)
    f.writelines("%s\n\n" % msg)
    f.close()

if __name__ == "__main__":
    generated_images_dir = './results/scagan_isnet_final/test_400/images'
    test(generated_images_dir)

