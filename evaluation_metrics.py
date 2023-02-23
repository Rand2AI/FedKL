import lpips, os, cv2, math
from skimage.metrics import structural_similarity
import numpy as np

import torch

class Quantitation(object):
    def __init__(self):
        self.lpips_vgg = lpips.LPIPS(net='vgg', spatial=False).cuda()
        self.lpips_alex = lpips.LPIPS(net='alex', spatial=False).cuda()

    @staticmethod
    def MSE(x, y):
        """
        @:keyword: smaller, better
        """
        return np.sqrt(np.mean((x.astype(np.float) - y.astype(np.float))**2))

    @staticmethod
    def PSNR(x, y, factor=255.0):
        """
        @:keyword: higher, better
        """
        mse = np.sqrt(np.mean((x.astype(np.float) - y.astype(np.float)) ** 2))
        return 10 * math.log10(factor ** 2 / mse)

    def LPIPS(self, x, y):
        """
        @:keyword: smaller, better
        """
        x_torch = torch.tensor(x).permute(2, 0, 1).unsqueeze(dim=0).cuda()
        y_torch = torch.tensor(y).permute(2, 0, 1).unsqueeze(dim=0).cuda()
        with torch.no_grad():
            lpips_score_vgg = self.lpips_vgg(x_torch, y_torch).squeeze().item()
            lpips_score_alex = self.lpips_alex(x_torch, y_torch).squeeze().item()
        return lpips_score_vgg, lpips_score_alex

    @staticmethod
    def SSIM( x, y):
        """
        @:keyword: higher, better
        """
        return structural_similarity(x, y, multichannel=True)

    def ASR(self):
        """
        @:keyword: We consider images as being successfully reconstructed if SSIM ≥ 0.6
        @:keyword: smaller, better. percentage
        """
        pass

def DLG_GRNN():
    evaluator = Quantitation()
    method = 'GRNN' # DLG GRNN
    network = 'res18'  # lenet res20 res18
    # datasets = ['mnist', 'cifar10', 'cifar100']
    datasets = ['cifar10', 'cifar100', 'imagenet']
    for dataset in datasets:
        root_paht = f'./{method}/{network}'
        mse = []
        psnr = []
        lpips_vgg = []
        lpips_alex = []
        ssim = []
        mse_kl = []
        psnr_kl = []
        lpips_vgg_kl = []
        lpips_alex_kl = []
        ssim_kl = []
        for d in os.listdir(root_paht):
            if f"-{dataset}-" in d:
                if d.endswith('-no_kl'):
                    print(d)
                    model_path = f"{root_paht}/{d}/saved_img/"
                    for idx in range(20):
                        try:
                            true_img_name = os.listdir(f"{model_path}/true_data/exp-{str(idx).zfill(6)}/")[0]
                            fake_img_name = os.listdir(f"{model_path}/fake_data/exp-{str(idx).zfill(6)}/")[0]
                            true_img = cv2.imread(f"{model_path}/true_data/exp-{str(idx).zfill(6)}/{true_img_name}")
                            fake_img = cv2.imread(f"{model_path}/fake_data/exp-{str(idx).zfill(6)}/{fake_img_name}")
                        except FileNotFoundError:
                            true_img_name = os.listdir(f"{model_path}/true_data/exp{str(idx).zfill(4)}/")[0]
                            fake_img_name = os.listdir(f"{model_path}/fake_data/exp{str(idx).zfill(4)}/")[0]
                            true_img = cv2.imread(f"{model_path}/true_data/exp{str(idx).zfill(4)}/{true_img_name}")
                            fake_img = cv2.imread(f"{model_path}/fake_data/exp{str(idx).zfill(4)}/{fake_img_name}")
                        mse.append(evaluator.MSE(true_img, fake_img))
                        psnr.append(evaluator.PSNR(true_img, fake_img))
                        lpips_score = evaluator.LPIPS(true_img, fake_img)
                        lpips_vgg.append(lpips_score[0])
                        lpips_alex.append(lpips_score[1])
                        ssim.append(evaluator.SSIM(true_img, fake_img))
                elif d.endswith('-kl'):
                    print(d)
                    model_path = f"{root_paht}/{d}/saved_img/"
                    for idx in range(20):
                        try:
                            true_img_name = os.listdir(f"{model_path}/true_data/exp-{str(idx).zfill(6)}/")[0]
                            fake_img_name = os.listdir(f"{model_path}/fake_data/exp-{str(idx).zfill(6)}/")[0]
                            true_img = cv2.imread(f"{model_path}/true_data/exp-{str(idx).zfill(6)}/{true_img_name}")
                            fake_img = cv2.imread(f"{model_path}/fake_data/exp-{str(idx).zfill(6)}/{fake_img_name}")
                        except FileNotFoundError:
                            true_img_name = os.listdir(f"{model_path}/true_data/exp{str(idx).zfill(4)}/")[0]
                            fake_img_name = os.listdir(f"{model_path}/fake_data/exp{str(idx).zfill(4)}/")[0]
                            true_img = cv2.imread(f"{model_path}/true_data/exp{str(idx).zfill(4)}/{true_img_name}")
                            fake_img = cv2.imread(f"{model_path}/fake_data/exp{str(idx).zfill(4)}/{fake_img_name}")
                        mse_kl.append(evaluator.MSE(true_img, fake_img))
                        psnr_kl.append(evaluator.PSNR(true_img, fake_img))
                        lpips_score = evaluator.LPIPS(true_img, fake_img)
                        lpips_vgg_kl.append(lpips_score[0])
                        lpips_alex_kl.append(lpips_score[1])
                        ssim_kl.append(evaluator.SSIM(true_img, fake_img))
        print(f"{method} {network} {dataset}")
        mse_mean = format(np.round(np.mean(mse), 2), '.2f')
        psnr_mean = format(np.round(np.mean(psnr), 2), '.2f')
        lpips_vgg_mean = format(np.round(np.mean(lpips_vgg), 2), '.2f')
        lpips_alex_mean = format(np.round(np.mean(lpips_alex), 2), '.2f')
        ssim_mean = format(np.round(np.mean(ssim), 3), '.3f')
        mse_kl_mean = format(np.round(np.mean(mse_kl), 2), '.2f')
        psnr_kl_mean = format(np.round(np.mean(psnr_kl), 2), '.2f')
        lpips_vgg_kl_mean = format(np.round(np.mean(lpips_vgg_kl), 2), '.2f')
        lpips_alex_kl_mean = format(np.round(np.mean(lpips_alex_kl), 2), '.2f')
        ssim_kl_mean = format(np.round(np.mean(ssim_kl), 3), '.3f')
        print(f"{mse_mean} & {mse_kl_mean} & {psnr_mean} & {psnr_kl_mean} & {lpips_vgg_mean} & {lpips_vgg_kl_mean} & {lpips_alex_mean} & {lpips_alex_kl_mean} & {ssim_mean} & {ssim_kl_mean}")

def GGL():
    root_paht = f'./GGL/'
    evaluator = Quantitation()
    mse = []
    psnr = []
    lpips_vgg = []
    lpips_alex = []
    ssim = []
    mse_kl = []
    psnr_kl = []
    lpips_vgg_kl = []
    lpips_alex_kl = []
    ssim_kl = []
    for d in os.listdir(root_paht):
        if d.endswith('-no_kl'):
            print(d)
            model_path = f"{root_paht}/{d}/"
            for idx in range(99,34099, 1000):
                for i in range(5):
                    true_img = cv2.imread(f"{model_path}/{idx}_true_{i}.png")
                    fake_img = cv2.imread(f"{model_path}/{idx}_out_{i}.png")
                    mse.append(evaluator.MSE(true_img, fake_img))
                    psnr.append(evaluator.PSNR(true_img, fake_img))
                    lpips_score = evaluator.LPIPS(true_img, fake_img)
                    lpips_vgg.append(lpips_score[0])
                    lpips_alex.append(lpips_score[1])
                    ssim.append(evaluator.SSIM(true_img, fake_img))
        elif d.endswith('-kl'):
            print(d)
            model_path = f"{root_paht}/{d}"
            for idx in range(99,37099, 1000):
                for i in range(5):
                    true_img = cv2.imread(f"{model_path}/{idx}_true_{i}.png")
                    fake_img = cv2.imread(f"{model_path}/{idx}_out_{i}.png")
                    mse_kl.append(evaluator.MSE(true_img, fake_img))
                    psnr_kl.append(evaluator.PSNR(true_img, fake_img))
                    lpips_score = evaluator.LPIPS(true_img, fake_img)
                    lpips_vgg_kl.append(lpips_score[0])
                    lpips_alex_kl.append(lpips_score[1])
                    ssim_kl.append(evaluator.SSIM(true_img, fake_img))
    mse_mean = format(np.round(np.mean(mse), 2), '.2f')
    psnr_mean = format(np.round(np.mean(psnr), 2), '.2f')
    lpips_vgg_mean = format(np.round(np.mean(lpips_vgg), 2), '.2f')
    lpips_alex_mean = format(np.round(np.mean(lpips_alex), 2), '.2f')
    ssim_mean = format(np.round(np.mean(ssim), 3), '.3f')
    mse_kl_mean = format(np.round(np.mean(mse_kl), 2), '.2f')
    psnr_kl_mean = format(np.round(np.mean(psnr_kl), 2), '.2f')
    lpips_vgg_kl_mean = format(np.round(np.mean(lpips_vgg_kl), 2), '.2f')
    lpips_alex_kl_mean = format(np.round(np.mean(lpips_alex_kl), 2), '.2f')
    ssim_kl_mean = format(np.round(np.mean(ssim_kl), 3), '.3f')
    print(f"{mse_mean} & {mse_kl_mean} & {psnr_mean} & {psnr_kl_mean} & {lpips_vgg_mean} & {lpips_vgg_kl_mean} & {lpips_alex_mean} & {lpips_alex_kl_mean} & {ssim_mean} & {ssim_kl_mean}")

def white_black():
    evaluator = Quantitation()

    white_32 = np.ones((32,32,3), dtype=np.uint8)*255
    black_32 = np.zeros((32,32,3), dtype=np.uint8)
    white_256 = np.ones((256,256,3), dtype=np.uint8)*255
    black_256 = np.zeros((256,256,3), dtype=np.uint8)

    mse_32 = evaluator.MSE(white_32, black_32)
    mse_256 = evaluator.MSE(white_256, black_256)

    psnr_32 = evaluator.PSNR(white_32, black_32)
    psnr_256 = evaluator.PSNR(white_256, black_256)

    lpips_vgg_32 = evaluator.LPIPS(white_32, black_32)[0]
    lpips_vgg_256 = evaluator.LPIPS(white_256, black_256)[0]

    lpips_alex_32 = evaluator.LPIPS(white_32, black_32)[1]
    lpips_alex_256 = evaluator.LPIPS(white_256, black_256)[1]

    ssim_32 = evaluator.SSIM(white_32, black_32)
    ssim_256 = evaluator.SSIM(white_256, black_256)

    print(f"32: {format(mse_32, '.2f')} & {format(psnr_32, '.2f')} & {format(lpips_vgg_32, '.2f')} & {format(lpips_alex_32, '.2f')} & {format(ssim_32, '.3f')}")
    print(f"256: {format(mse_256, '.2f')} & {format(psnr_256, '.2f')} & {format(lpips_vgg_256, '.2f')} & {format(lpips_alex_256, '.2f')} & {format(ssim_256, '.3f')}")
if __name__=='__main__':
    # DLG_GRNN()
    # GGL()
    white_black()