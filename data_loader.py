import os
import random
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import scipy.ndimage
import torch


class ImageFolder(data.Dataset):
    def __init__(self, root, image_size=None, mode='train', augmentation_prob=0.4, exp_or_sim='sim', image_type='8bits',config=None):
        if config is None:
            raise ValueError("check the config in the dataloader")
        self.config = config

        self.root = root  # 'xxxxxxx/trian_valid' or 'xxxxxxx/test'
        self.mode = mode
        print(f'\n=============> Loading data from {self.root} for {mode}. ==========>\n')
        self.image_type = image_type

        train_input_list = []
        train_gt_list = []
        valid_input_list = []
        valid_gt_list = []
        test_input_list = []
        test_gt_list = []

        if self.mode == 'train' or self.mode == 'valid':
            folders = config.selected_train_valid_fold  # ['2nanoholes', '3nanoholes', ..., '10nanoholes']
        elif self.mode == 'test':
            folders = config.selected_test_fold

        for fold in folders:
            fold_input = os.path.join(root, fold, 'img')
            fold_gt = os.path.join(root, fold, 'mask')

            # get the name of fold
            fold_name = fold.split('/')[-1]

            num_data_in_fold = len(os.listdir(fold_input))
            upper_limit = 10000000  # determine the amount of training dataset

            if self.mode == 'train' or self.mode == 'valid':
                valid_rate = config.valid_rate  # e.g., 10%
                valid_num = int(valid_rate * num_data_in_fold)
                file_list = os.listdir(fold_input)
                random.shuffle(file_list)
                # for i, item in enumerate(sorted(os.listdir(fold_input))):
                for i, item in enumerate(file_list):  # make sure the train and valid is IID
                    if i < valid_num:
                        valid_input_list.append(os.path.join(fold_input, item))
                        valid_gt_list.append(os.path.join(fold_gt, item))
                    elif i >= valid_num and i < upper_limit:
                        train_input_list.append(os.path.join(fold_input, item))
                        train_gt_list.append(os.path.join(fold_gt, item))
                    else:
                        break


            if self.mode == 'test':
                for item in range(num_data_in_fold):
                    test_input_list.append(os.path.join(fold_input, str(item)+'.tif'))
                    test_gt_list.append(os.path.join(fold_gt, str(item)+'.tif'))

        if self.mode == 'train':
            self.image_paths = train_input_list
            self.GT_paths = train_gt_list
        elif self.mode == 'valid':
            self.image_paths = valid_input_list
            self.GT_paths = valid_gt_list
        elif self.mode == 'test':
            self.image_paths = test_input_list
            self.GT_paths = test_gt_list

        self.image_size = self.config.image_size
        self.config = config
        self.RotationDegree = [0, 90, 180, 270]
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        #print(f'index {index}')
        image_path = self.image_paths[index]
        GT_path = self.GT_paths[index]
        image_type = self.image_type

        if image_type == '16bits':
            image = Image.open(image_path)
            GT = Image.open(GT_path)
            ResizeRange = 256

            # ============================ try float32 ========================== #
            image_array = np.asarray(image).astype(np.float32)
            GT_array   = np.asarray(GT).astype(np.float32)
            # =================================================================== #

            zoom_ratio_x = ResizeRange/image_array.shape[0]
            zoom_ratio_y = ResizeRange/image_array.shape[1]
            image_array_resize = scipy.ndimage.zoom(image_array, zoom=(zoom_ratio_x, zoom_ratio_y), order=1)

            GT_zoom_ratio_x = ResizeRange / GT_array.shape[0]
            GT_zoom_ratio_y = ResizeRange / GT_array.shape[1]
            GT_array_resize = scipy.ndimage.zoom(GT_array, zoom=(GT_zoom_ratio_x, GT_zoom_ratio_y), order=1)

            image_increase = image_array_resize
            GT_increase = GT_array_resize
            if (self.mode == 'train'):

                if random.random() < 0.5:
                    image_increase = np.rot90(image_array_resize)
                    GT_increase = np.rot90(GT_array_resize)

                if random.random() < 0.5:
                    image_increase= np.fliplr(image_increase)
                    GT_increase= np.fliplr(GT_increase)

                if random.random() < 0.5:
                    image_increase = np.flipud(image_increase)
                    GT_increase = np.flipud(GT_increase)

            min_value = np.min(image_increase)
            max_value = np.max(image_increase)

            normalized_image_array = (image_increase - min_value) / (max_value - min_value)
            image_array_nor = (normalized_image_array * 2) - 1
            image_trans = torch.tensor(image_array_nor.copy()).unsqueeze(0)

            GT_trans = torch.tensor(GT_increase.copy()).unsqueeze(0)
            GT_trans = GT_trans/255


        if image_type =='8bits':
            image = Image.open(image_path)
            GT    = Image.open(GT_path)

            Transform = []
            aspect_ratio = image.size[1] / image.size[0]

            #  ===============  Resize the image ============= #
            # aspect_ratio = image.size[1] / image.size[0]
            # ResizeRange = 256
            # t_resize = T.Resize((int(ResizeRange * aspect_ratio), ResizeRange))
            # image = t_resize(image)
            # GT = t_resize(GT)

            # ==========================================================================#

            if (self.mode == 'train'):
                if self.config.rotate:
                    RotationDegree = random.randint(0, 3)
                    RotationDegree = self.RotationDegree[RotationDegree]
                    if (RotationDegree == 90) or (RotationDegree == 270):
                        aspect_ratio = 1 / aspect_ratio

                    Transform.append(T.RandomRotation((RotationDegree, RotationDegree)))
            # ==========================================================================#

                if random.random() < 0.5:
                    image = F.hflip(image)
                    GT = F.hflip(GT)

                if random.random() < 0.5:
                    image = F.vflip(image)
                    GT = F.vflip(GT)

                # if random.random() < 0.5:
                #     angle = random.uniform(-30, 30)
                #     image = F.rotate(image, angle)
                #     GT = F.rotate(GT, angle)

                # if random.random() < 0.5:
                #     brightness_factor = random.uniform(0.8, 1.2)
                #     contrast_factor = random.uniform(0.8, 1.2)
                #     saturation_factor = random.uniform(0.8, 1.2)
                #     image = F.adjust_brightness(image, brightness_factor)
                #     image = F.adjust_contrast(image, contrast_factor)
                #     image = F.adjust_saturation(image, saturation_factor)

            if self.config.center_crop:
                CropRange = random.randint(256, 256)
                crop_operation = T.CenterCrop((int(CropRange * aspect_ratio), CropRange))
                image = crop_operation(image)

            # Transform.append(T.Resize((256, 256)))
            Transform.append(T.ToTensor())
            Transform = T.Compose(Transform)
            image_trans = Transform(image)

            # Check is GT is 0_1 or 0_255
            GT_array = np.array(GT)
            if np.max(GT_array) <= 1.1 :
                GT_trans = torch.from_numpy(GT_array).float()
            else:

                GT_trans = Transform(GT)
            # Ensure GT is binary
            GT_trans = (GT_trans > 0.5).float()

            GT_trans = GT_trans.unsqueeze(0)

            Norm_ = T.Normalize(mean=0.5, std=0.5)
            image_trans = Norm_(image_trans)
            if self.config.model_type in ['Transformer_UNet'] and self.config.encoder_only:
                h, w = GT_trans.shape[-2], GT_trans.shape[-1]
                gh, gw = self.config.output_hw, self.config.output_hw
                grid_height = h // gh
                grid_width = w // gw
                centers = []
                for i in range(gh):
                    for j in range(gw):
                        # 计算每个网格的中心点坐标
                        center_y = i * grid_height + grid_height // 2
                        center_x = j * grid_width + grid_width // 2
                        centers.append((center_y, center_x))
                GT_9values = []
                for center in centers:
                    GT_9values.append(GT_trans[0, center[0], center[1]])
                GT_trans = torch.tensor(GT_9values)



            #     noise = 0.01 * torch.randn_like(image_trans)
            #     image_trans += noise
            #     GT_trans = image_trans

        return image_trans, GT_trans, image_path, GT_path

# ================== debug for seeing the image_trans ========== #
    # import matplotlib.pyplot as plt
    # import numpy as np

    # arr = image_trans.cpu().detach().numpy()
    # # 去掉批次维度
    # arr = arr.squeeze(0)  # 现在 arr 的形状是 (64, 64)

    # plt.figure(figsize=(10, 8))

    # im = plt.imshow(arr, cmap='gray', vmin=-1, vmax=1)  --- for image_trans 经过了Norm 到了-1,1

    # plt.colorbar(im)
    # plt.title("Visualization of image_trans")

    # plt.show()

    # print(f"Min value: {arr.min()}, Max value: {arr.max()}")
    # print(f"Mean: {arr.mean()}, Std: {arr.std()}")

# ================== debug for seeing the GT_trans ==========
    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # # 将 GT_trans 转换为 numpy 数组
    # arr = GT_trans.cpu().detach().numpy()
    #
    # # 去掉批次维度（如果有的话）
    # arr = arr.squeeze()
    #
    # # 创建图形
    # plt.figure(figsize=(10, 8))
    #
    # # 使用 imshow 显示图像数据
    # # 由于 GT_trans 是 0-1 分布，我们使用 vmin=0, vmax=1
    # im = plt.imshow(arr, cmap='gray', vmin=0, vmax=1)
    #
    # # 添加颜色条
    # plt.colorbar(im)
    #
    # # 设置标题
    # plt.title("Visualization of GT_trans")
    #
    # # 显示图形
    # plt.show()
    #
    # # 打印统计信息
    # print(f"Min value: {arr.min()}, Max value: {arr.max()}")
    # print(f"Mean: {arr.mean()}, Std: {arr.std()}")
    #
    # # 打印唯一值及其计数
    # unique, counts = np.unique(arr, return_counts=True)
    # print("Unique values and their counts:")
    # for value, count in zip(unique, counts):
    #     print(f"Value {value}: {count} occurrences")

# ================== debug for seeing the input GT / Image ========== #

    # import matplotlib.pyplot as plt
    # import numpy as np
    # from PIL import Image
    #
    # # 假设 GT 是您通过 Image.open(GT_path) 打开的图像
    # # 如果 GT 还不是 numpy 数组，我们需要转换它
    # gt_array = np.array(GT)

    # plt.figure(figsize=(10, 8))
    #
    # # 检查图像是彩色还是灰度
    # if len(gt_array.shape) == 3 and gt_array.shape[2] == 3:
    #     # 彩色图像
    #     im = plt.imshow(gt_array)
    #     plt.title("GT Image (Color)")
    # else:
    #     # 灰度图像
    #     im = plt.imshow(gt_array, cmap='gray')
    #     plt.title("GT Image (Grayscale)")
    #

    # plt.colorbar(im)

    # plt.show()

    # print(f"Image shape: {gt_array.shape}")
    # print(f"Data type: {gt_array.dtype}")
    # print(f"Min value: {gt_array.min()}, Max value: {gt_array.max()}")


    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)


def get_loader(image_path, image_type, image_size, batch_size, num_workers=2, mode='train', augmentation_prob=0.4, exp_or_sim=None,
               config=None):
    """Builds and returns Dataloader."""

    dataset = ImageFolder(root=image_path, image_type=image_type,image_size=image_size, mode=mode, augmentation_prob=augmentation_prob,
                          exp_or_sim=exp_or_sim, config=config)
    if mode == 'train':
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)
    elif mode == 'test':
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=num_workers)
    elif mode == 'valid':
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=num_workers)

    return data_loader
