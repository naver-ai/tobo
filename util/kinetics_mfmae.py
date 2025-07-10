import os
import random
import pickle

from decord import VideoReader, cpu

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F



class PairedRandomResizedCrop:
    def __init__(
        self,
        hflip_p=0.5,
        size=(224, 224),
        scale=(0.5, 1.0),
        ratio=(3./4., 4./3.),
        interpolation=F.InterpolationMode.BICUBIC,
        framewise_flip=False
    ):
        self.hflip_p = hflip_p
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.framewise_flip = framewise_flip

    def __call__(self, np_RGB_img_1, np_RGB_img_2):
        # Convert numpy images to PIL Images
        pil_RGB_img_1 = F.to_pil_image(np_RGB_img_1)
        pil_RGB_img_2 = F.to_pil_image(np_RGB_img_2)

        i, j, h, w = transforms.RandomResizedCrop.get_params(
            pil_RGB_img_1, scale=self.scale, ratio=self.ratio
        )
        # Apply the crop on both images
        cropped_img_1 = F.resized_crop(pil_RGB_img_1,
                                       i, j, h, w,
                                       size=self.size,
                                       interpolation=self.interpolation)
        cropped_img_2 = F.resized_crop(pil_RGB_img_2,
                                       i, j, h, w,
                                       size=self.size,
                                       interpolation=self.interpolation)
        if self.framewise_flip:
            if random.random() < self.hflip_p:
                cropped_img_1 = F.hflip(cropped_img_1)
            if random.random() < self.hflip_p:
                cropped_img_2 = F.hflip(cropped_img_2)
        else:
            if random.random() < self.hflip_p:
                cropped_img_1 = F.hflip(cropped_img_1)
                cropped_img_2 = F.hflip(cropped_img_2)

        return cropped_img_1, cropped_img_2



class RandomResizedCrop:
    def __init__(
        self,
        hflip_p=0.5,
        size=(224, 224),
        scale=(0.5, 1.0),
        ratio=(3./4., 4./3.),
        interpolation=F.InterpolationMode.BICUBIC,
        framewise_flip=False,
        unpaired_rrc=False,
        random_rotation=False,
        rotation_range=45,
    ):
        self.hflip_p = hflip_p
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.framewise_flip = framewise_flip
        self.unpaired_rrc = unpaired_rrc
        self.random_rotation = random_rotation
        self.rotation_range = rotation_range

    def __call__(self, np_RGB_img_1, np_RGB_img_2):
        # Convert numpy images to PIL Images
        pil_RGB_img_1 = F.to_pil_image(np_RGB_img_1)
        pil_RGB_img_2 = F.to_pil_image(np_RGB_img_2)


        if self.unpaired_rrc:
            i2, j2, h2, w2 = transforms.RandomResizedCrop.get_params(
                pil_RGB_img_1, scale=self.scale, ratio=self.ratio
            )

            cropped_img_2 = F.resized_crop(pil_RGB_img_2,
                                           i2, j2, h2, w2,
                                           size=self.size,
                                           interpolation=self.interpolation)
        else:
            cropped_img_2 = F.resized_crop(pil_RGB_img_2,
                                           i, j, h, w,
                                           size=self.size,
                                           interpolation=self.interpolation)


        if self.framewise_flip:
            if random.random() < self.hflip_p:
                cropped_img_1 = F.hflip(cropped_img_1)
            if random.random() < self.hflip_p:
                cropped_img_2 = F.hflip(cropped_img_2)
        else:
            if random.random() < self.hflip_p:
                cropped_img_1 = F.hflip(cropped_img_1)
                cropped_img_2 = F.hflip(cropped_img_2)

        return cropped_img_1, cropped_img_2



class MultiRandomResizedCrop:
    def __init__(
        self,
        hflip_p=0.5,
        size=(224, 224),
        scale=(0.5, 1.0),
        ratio=(3./4., 4./3.),
        interpolation=F.InterpolationMode.BICUBIC,
        framewise_flip=False,
        unpaired_rrc=False,
    ):
        self.hflip_p = hflip_p
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.framewise_flip = framewise_flip
        self.unpaired_rrc = unpaired_rrc

    def __call__(self, list_of_np_imgs):
        # Convert numpy images to PIL Images

        # pil_RGB_img_1 = F.to_pil_image(list_of_np_imgs[0])
        # i, j, h, w = transforms.RandomResizedCrop.get_params(
        #     pil_RGB_img_1, scale=self.scale, ratio=self.ratio
        # )

        hflip_prob = random.random()

        list_output = []
        for i, img in enumerate(list_of_np_imgs):
            pil_img = F.to_pil_image(img)

            # Apply the crop on both images
            if self.unpaired_rrc:
                i, j, h, w = transforms.RandomResizedCrop.get_params(
                                           pil_img, scale=self.scale, ratio=self.ratio
                                           )
                cropped_img = F.resized_crop(pil_img,
                                           i, j, h, w,
                                           size=self.size,
                                           interpolation=self.interpolation)
            else:
                if i == 0:
                    i, j, h, w = transforms.RandomResizedCrop.get_params(
                                           pil_img, scale=self.scale, ratio=self.ratio
                                           )

                cropped_img = F.resized_crop(pil_img,
                                           i, j, h, w,
                                           size=self.size,
                                           interpolation=self.interpolation)
            if self.framewise_flip:
                if random.random() < self.hflip_p:
                    cropped_img = F.hflip(cropped_img)
            else:
                if hflip_prob < self.hflip_p:
                    cropped_img = F.hflip(cropped_img)

            list_output.append(cropped_img)

        return list_output


class MultiKinetics(Dataset):
    def __init__(
        self,
        root,
        max_distance=48,
        repeated_sampling=2,
        num_frames=2,
        framewise_flip=False,
        unpaired_rrc=False,
        basic_transform=None
    ):
        super().__init__()
        self.root = root
        with open(
            os.path.join(self.root, "labels", f"label_1.0.pickle"), "rb"
        ) as f:
            self.samples = pickle.load(f)

        self.transforms = MultiRandomResizedCrop(framewise_flip=framewise_flip,
                                                 unpaired_rrc=unpaired_rrc)
        if basic_transform is None:
            self.basic_transform = transforms.Compose(
                [
                 # Stack(roll=False),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            )
        else:
            self.basic_transform = basic_transform


        self.max_distance = max_distance
        self.repeated_sampling = repeated_sampling
        self.num_frames = num_frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = os.path.join(self.root, self.samples[index][1])
        vr = VideoReader(sample, num_threads=1, ctx=cpu(0))

        # src_images = []
        # tgt_images = []
        list_images = []
        for i in range(self.repeated_sampling):
            images = self.load_frames(vr)
            images = self.transform(images)
            images = torch.stack(images, dim=0)
            list_images.append(images)
        output = torch.stack(list_images, dim=0)
        return output, 0

    def load_frames(self, vr):
        # handle temporal segments
        seg_len = len(vr)
        least_frames_num = self.max_distance + 1
        if seg_len >= least_frames_num:
            idx_cur = random.randint(0, seg_len - least_frames_num)
            interval = random.randint(1, self.max_distance // (self.num_frames - 1))
            # idx_fut = idx_cur + interval
            list_idx = [idx_cur + i * interval for i in range(0, self.num_frames)]
        elif seg_len < self.num_frames:
            print('setting error')
            list_idx = []
            for i in range(self.num_frames - seg_len):
                list_idx.append(random.randint(0, seg_len - 1))
            for i in range(seg_len):
                list_idx.append(i)
        elif seg_len == self.num_frames:
            list_idx = [i for i in range(0, self.num_frames)]
        else:
            interval = random.randint(1, (seg_len - 1) // (self.num_frames - 1))
            idx_cur = random.randint(0, seg_len - 1 - interval * (self.num_frames - 1))
            # print(seg_len, idx_cur, interval, self.num_frames)
            list_idx = [idx_cur + i * interval for i in range(0, self.num_frames)]

        frames = [vr[i].asnumpy() for i in list_idx]

        return frames

    def transform(self, images):
        images = self.transforms(images)

        list_output = []
        for image in images:
            image = self.basic_transform(image)
            list_output.append(image)
        return list_output


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):

        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2),
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)
