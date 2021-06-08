#
# 2021 (c) by **Mirko Lantieri**
# All rights reserved.
#
# activator.py : script responsable for the activation maps
# The file contains the respective methods:
# *get_args* : runs the script when parsing the arguments via terminal


# Importing libraries
import argparse

import cv2
import numpy as np
import torch
import os
from torchvision import models
from cam.ablation_cam import AblationCAM
from cam.grad_cam import GradCAM
from cam.grad_cam_pp import GradCAMPlusPlus
from cam.guided_backprop import GuidedBackpropReLUModel
from cam.score_cam import ScoreCAM
from cam.xgrad_cam import XGradCAM
from src.models.model import AlexNet
from src.helper.utils import deprocess_image, preprocess_image, show_cam_on_image


def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--path', default=False, help='Path of the pretrained model')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image', type=str, default=False, 
                        help='Input image path')
    parser.add_argument('--model', type=str, default=False, help='path of the stored model')
    parser.add_argument('--method', type=str, default='gradcam',
                        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')
    parser.add_argument('--case', type=str, default=False, help='number of the case')
    parser.add_argument('--plane', type=str, default=False, help='plane of the case')
    parser.add_argument('--metric', type=str, default=1, help='choose the metric based on accuracy - weighted utility - auc', choices=['1', '2', '3'])
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python activation.py --image <path_to_image> --path <path_pretrained_model> --case <no_case> --plane <axial or sagittal or coronal>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    # Select one the available method from the parsed arguments

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    #mrnet = torch.load(args.model)
    model = models.resnet18(pretrained=True)
    #model.load_state_dict(mrnet, strict=False)
    
    """
    Choose the target layer you want to compute the visualization for.
    Usually this will be the last convolutional layer in the model.
    Some common choices can be:
        Resnet18 and 50: model.layer4[-1]
        VGG, densenet161: model.features[-1]
        mnasnet1_0: model.layers[-1]
        You can print the model to help chose the layer
        print(model) 
    """

    target_layer = model.layer4[-1]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    cam = methods[args.method](model=model,
                               target_layer=target_layer,
                               use_cuda=args.cuda)

    rgb_img = cv2.imread(args.image, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category)

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)

    gb_model = GuidedBackpropReLUModel(
        model=model, use_cuda=args.cuda)
    gb = gb_model(input_tensor, target_category=target_category)

    cam_mask = cv2.merge(
        [grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    if args.metric == '2':
        cv2.imwrite(f'./images/activations/wu/case{args.case}{args.plane}_{args.method}.jpg', cam_image)
    elif args.metric == '3':
        cv2.imwrite(f'./images/activations/auc/case{args.case}{args.plane}_{args.method}.jpg', cam_image)
    else:
        cv2.imwrite(f'./images/activations/acc/case{args.case}{args.plane}_{args.method}.jpg', cam_image)

    # remove the comment to activate also the guided backpropagation generated from the model
    #cv2.imwrite(f'./images/activations/case{args.case}{args.plane}_{args.method}_gb.jpg', gb)
    #cv2.imwrite(f'./images/activations/case{args.case}{args.plane}_{args.method}_cam_gb.jpg', cam_gb)
