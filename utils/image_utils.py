import torch
import torch.nn.functional as F

def myPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1) + 1e-8
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def batch_PSNR(img1, img2, average=True):
    PSNR = []
    for im1, im2 in zip(img1, img2):
        psnr = myPSNR(im1, im2)
        PSNR.append(psnr)
    return sum(PSNR)/len(PSNR) if average else sum(PSNR)

def mySSIM(tar_img, prd_img):
    tar_mean = F.avg_pool2d(tar_img, kernel_size=3, stride=1, padding=0)
    prd_mean = F.avg_pool2d(prd_img, kernel_size=3, stride=1, padding=0)

    tar_var = F.avg_pool2d(tar_img**2, kernel_size=3, stride=1, padding=0) - tar_mean**2
    prd_var = F.avg_pool2d(prd_img**2, kernel_size=3, stride=1, padding=0) - prd_mean**2

    covar = F.avg_pool2d(tar_img * prd_img, kernel_size=3, stride=1, padding=0) - tar_mean * prd_mean

    C1 = 0.01**2
    C2 = 0.03**2

    ssim = (2 * tar_mean * prd_mean + C1) * (2 * covar + C2) / ((tar_mean**2 + prd_mean**2 + C1) * (tar_var + prd_var + C2))

    return ssim.mean()

def batch_SSIM(img1, img2, average=True):
    SSIM = []
    for im1, im2 in zip(img1, img2):
        ssim = mySSIM(im1, im2)
        SSIM.append(ssim)
    return sum(SSIM)/len(SSIM) if average else sum(SSIM)

