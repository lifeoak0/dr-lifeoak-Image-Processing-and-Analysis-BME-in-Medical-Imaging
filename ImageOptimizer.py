import numpy as np #这里使用到了numpy skimage（scikit-image） python库提供一系列图像处理功能，通常协同numpy和scipy工作，作为scipy生态一部分此库提供一系列图像处理工作的功能，包括但不限于凸显分割，几何变换，分析滤波，特征检验    
from skimage import io, exposure, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma

def process_image(image_path):
    # 读取图像,放入个人图像本地路径
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

#代码总体思路：
#首先定义函数def  process_image(image_path  使用io.imread读取图像，然后将image转化为灰度图像as_gray=true, 然后将img-as-float将图像转化为浮点数方便后续处理。
#exposure.equalize_adapthist增强图像对比度，
#噪声估计noise estimate和denoiseestimate_sigma 用于估计图像的噪声水平。 denoise_nl_means 实现基于非局部均值的去噪算法  其中‘h’‘patch—size path-distance用来调整参数改变去噪过程’
#最后 返回处理后的图像
