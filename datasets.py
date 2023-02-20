# Albumentations
import albumentations
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset,DataLoader
# vision utils
import cv2

from config import CFG


def get_train_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(CFG.img_size,CFG.img_size,always_apply=True),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=120, p=0.8),
            #albumentations.RandomBrightnessContrast(limit=(0.09, 0.6), p=0.5),
            albumentations.Normalize(mean = CFG.MEAN, std = CFG.STD),
            ToTensorV2(p=1.0)
        ]
    )

def get_valid_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(CFG.img_size,CFG.img_size, always_apply=True),
            albumentations.Normalize(mean = CFG.MEAN, std = CFG.STD),
            ToTensorV2(p=1.0)
        ]
    )

#########################################

class EmotionDataset(Dataset):
    def __init__(self, data, transform=None):
        # Load csv data
        self.df = data
        self.image_arr = np.asarray(self.df.iloc[:, 0])
        self.label_arr = np.asarray(self.df.iloc[:, 1])
        # transform 여부
        self.transform = transform

    def __len__(self):
        self.data_len = len(self.label_arr)
        return self.data_len

    def __getitem__(self, index):
        row = self.image_arr[index]  # csv이미지파일명 index선언
        label_row = self.label_arr[index]  # label값에 대해 index선언
        image = cv2.imread(row)  # 해당 index에 대한 이미지 로드
        #image = cv2.imread(row , cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # color차원 부여 -> tensor자료형
        #image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, torch.tensor(label_row)
