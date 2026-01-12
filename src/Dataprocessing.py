from torchvision import transforms as T
from PIL import Image
import numpy as np
import cv2




class DRPreprocess:
    """Apply retina-specific preprocessing before Tensor conversion."""
    def __call__(self, img):
        img = self.circle_crop(img)
        img = self.apply_clahe(img)
        return img

    def circle_crop(self, pil_img):
        img = np.array(pil_img)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        radius = min(center)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)

        cropped = cv2.bitwise_and(img, img, mask=mask)
        return Image.fromarray(cropped)

    def apply_clahe(self, pil_img):
        img = np.array(pil_img)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        merged = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

        return Image.fromarray(enhanced)

def get_dr_transforms(img_size=224):

    train_transform = T.Compose([
        DRPreprocess(),                             # Retina preprocessing
        T.Resize(int(img_size * 1.15)),
        T.CenterCrop(img_size),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
        T.RandomApply([T.GaussianBlur(5)], p=0.3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    val_transform = T.Compose([
        DRPreprocess(),
        T.Resize(int(img_size * 1.15)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform
