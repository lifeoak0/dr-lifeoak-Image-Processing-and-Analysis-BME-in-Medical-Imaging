import numpy as np
from skimage import io, exposure, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma

def process_image(image_path):
    # 读取图像
    image = io.imread(image_path, as_gray=True)
    image = img_as_float(image)  # 将图像转换为浮点类型

    # 对比度增强
    image_contrast_enhanced = exposure.equalize_adapthist(image)

    # 估计图像噪声并去噪
    sigma_est = np.mean(estimate_sigma(image_contrast_enhanced))
    denoised_image = denoise_nl_means(image_contrast_enhanced, h=1.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=6)

    # 返回处理后的图像
    return denoised_image

# 使用示例
processed_image = process_image(r'T:\CT_MRI_IMAGE\GGG.jpg')  # 替换为你的图像文件路径

# 展示处理后的图像
io.imshow(processed_image)
io.show()
