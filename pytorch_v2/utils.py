import torch
import torch.nn.functional as F


def random_crop(img: torch.Tensor, padding: int, rng: torch.Generator) -> torch.Tensor:
    """
    对单张图像进行随机裁剪，保持原始大小。

    Args:
        img: (C, H, W)
        padding: int, padding size
        rng: torch.Generator 用于可控随机性

    Returns:
        cropped image of shape (C, H, W)
    """
    c, h, w = img.shape
    # 边缘填充
    padded = F.pad(
        img, pad=(padding, padding, padding, padding), mode="replicate"
    )  # (C, H+2p, W+2p)

    # 在 padding 区域中随机裁剪位置
    top = torch.randint(0, 2 * padding + 1, (1,), generator=rng).item()
    left = torch.randint(0, 2 * padding + 1, (1,), generator=rng).item()

    cropped = padded[:, top : top + h, left : left + w]
    return cropped


def batched_random_crop(
    batch_imgs: torch.Tensor, padding: int, seed: int = None
) -> torch.Tensor:
    """
    对一个 batch 的图像进行随机裁剪。

    Args:
        batch_imgs: (B, C, H, W)
        padding: int
        seed: 可选随机种子，便于可复现性

    Returns:
        cropped_imgs: (B, C, H, W)
    """
    B, C, H, W = batch_imgs.shape
    assert C == 3, "图像通道数必须为3"

    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    cropped_imgs = torch.stack(
        [random_crop(batch_imgs[i], padding=padding, rng=rng) for i in range(B)]
    )
    return cropped_imgs
