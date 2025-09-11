from torchvision import transforms


crop_example_t = transforms.Lambda(
    lambda x: transforms.functional.crop(x, top=250, left=380, height=200, width=300)
)


def get_train_transform(crop=None):
    if crop is not None:
        crop_t = transforms.Lambda(
            lambda x: transforms.functional.crop(
                x, top=crop[0], left=crop[1], height=crop[2], width=crop[3]
            )
        )
    else:
        crop_t = transforms.Lambda(lambda x: x)  # no crop

    return transforms.Compose(
        [
            # pre-process
            transforms.ToPILImage(),  # Image need HxWxC
            crop_t,
            # data augmentations
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
            # post-process
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_eval_transform(crop=None):
    if crop is not None:
        crop_t = transforms.Lambda(
            lambda x: transforms.functional.crop(
                x, top=crop[0], left=crop[1], height=crop[2], width=crop[3]
            )
        )
    else:
        crop_t = transforms.Lambda(lambda x: x)  # no crop

    return transforms.Compose(
        [
            # pre-process
            transforms.ToPILImage(),  # Image need HxWxC
            crop_t,
            # post-process
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
