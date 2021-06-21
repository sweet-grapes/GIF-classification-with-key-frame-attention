import os
import random
import imageio
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset
# new added
from PIL import Image
import torchvision.transforms.functional as F

# new added


class ResizedCrop:
    def __init__(self, crop_size, params):
        self.size = crop_size
        self.params = params

    def __call__(self, image):
        i, j, h, w = self.params
        image = F.resized_crop(image, i, j, h, w, self.size, Image.CUBIC)
        return image


class GIFDataset(Dataset):
    CLASS_INDEX_MAP = {
        '举重': 0,
        '乒乓球': 1,
        '云': 2,
        '仓鼠': 3,
        '企鹅': 4,
        '体操': 5,
        '健身': 6,
        '兔子': 7,
        '冰': 8,
        '冰川': 9,
        '冲浪': 10,
        '北极光': 11,
        '博美': 12,
        '吉娃娃': 13,
        '哈士奇': 14,
        '大象': 15,
        '射击': 16,
        '小浣熊': 17,
        '小黄鸭': 18,
        '山': 19,
        '山羊': 20,
        '帆船': 21,
        '手球': 22,
        '拳击': 23,
        '排球': 24,
        '日出': 25,
        '日落': 26,
        '星空': 27,
        '月亮': 28,
        '松鼠': 29,
        '柯基': 30,
        '棒球': 31,
        '森林': 32,
        '植物': 33,
        '沙漠': 34,
        '泰迪': 35,
        '海洋': 36,
        '海滩': 37,
        '海濑': 38,
        '游泳': 39,
        '湖': 40,
        '滑板': 41,
        '滑雪': 42,
        '瀑布': 43,
        '熊猫': 44,
        '狮子': 45,
        '猪': 46,
        '猫咪': 47,
        '猴子': 48,
        '瑜伽': 49,
        '秋天': 50,
        '篮球': 51,
        '网球': 52,
        '羽毛球': 53,
        '老虎': 54,
        '自行车': 55,
        '草原': 56,
        '袋鼠': 57,
        '豪猪': 58,
        '足球': 59,
        '跑步': 60,
        '跨栏': 61,
        '跳伞': 62,
        '跳水': 63,
        '跳舞': 64,
        '长颈鹿': 65,
        '雪山': 66,
        '雪貂': 67,
        '雾': 68,
        '鱼': 69,
        '鸟': 70,
        '龙卷风': 71,
        '龙猫': 72
    }

    def __init__(self, data_root):
        super().__init__()
        self.GIF_list = []
        for class_name in self.CLASS_INDEX_MAP:
            class_dir = os.path.join(data_root, class_name)
            for GIF_name in os.listdir(class_dir):
                GIF_path = os.path.join(class_dir, GIF_name)
                GIF_label = self.CLASS_INDEX_MAP[class_name]
                self.GIF_list.append((GIF_path, GIF_label))
        # random.shuffle(self.GIF_list)

    def __getitem__(self, idx):
        gif_path = self.GIF_list[idx][0]
        label = self.GIF_list[idx][1]

        frames = np.array(list(imageio.get_reader(gif_path)))

        image = frames[random.choice(range(frames.shape[0]))]

        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        image = image[:, :, :3]
        # new added
        crop_params = T.RandomResizedCrop.get_params(Image.fromarray(frames[0]), (0.8, 1), (3 / 4, 4 / 3))
        transform = T.Compose(
            [T.ToPILImage(),
             ResizedCrop((224, 224), crop_params),
             # T.Resize((224, 224)),
             T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        image = transform(image)

        return image, label

    def __len__(self):
        return len(self.GIF_list)


