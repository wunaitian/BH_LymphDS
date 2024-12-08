import os
import math
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torchvision import transforms
from BH_LymphDS.custom.utils import GradCAM, show_cam_on_image, center_crop_img
from BH_LymphDS.models.vits.swin_model import swin_base_patch4_window7_224


class ResizeTransform:
    def __init__(self, im_h: int, im_w: int):
        self.height = self.feature_size(im_h)
        self.width = self.feature_size(im_w)

    @staticmethod
    def feature_size(s):
        s = math.ceil(s / 4)  # PatchEmbed
        s = math.ceil(s / 2)  # PatchMerging1
        s = math.ceil(s / 2)  # PatchMerging2
        s = math.ceil(s / 2)  # PatchMerging3
        return s

    def __call__(self, x):
        result = x.reshape(x.size(0),
                           self.height,
                           self.width,
                           x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)

        return result


def swin_gradcam(model_path, data):

    img_size = 512
    assert img_size % 32 == 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = swin_base_patch4_window7_224()
    weights_path = os.path.join(model_path, 'BH_LymphDS.pth')
    weights_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(weights_dict['model_state_dict'], strict=True)
    model.to(device)

    target_layers = [model.norm]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    save_dir = r".\Output\GradCam"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True,
                  reshape_transform=ResizeTransform(im_h=img_size, im_w=img_size))

    dir_total = data
    img_dirs = [os.path.join(dir_total, dir_path) for dir_path in os.listdir(dir_total)]
    for img_dir in img_dirs:
        img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
        for img_path in img_paths:
            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img = Image.open(img_path).convert('RGB')
            img = np.array(img, dtype=np.uint8)
            img = center_crop_img(img, img_size)

            # [C, H, W]
            img_tensor = data_transform(img)
            # expand batch dimension
            # [C, H, W] -> [N, C, H, W]
            input_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)

            # target_category = 0
            # grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
            grayscale_cam = cam(input_tensor=input_tensor)

            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(img / 255., grayscale_cam, use_rgb=True)

            original_filename = os.path.basename(img_path)
            save_path = os.path.join(save_dir, original_filename)

            plt.imsave(save_path, visualization)






