import torchvision.models as models
import torch

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(models.vgg16(pretrained=True).features[:4].to(device).eval())
        blocks.append(models.vgg16(pretrained=True).features[4:9].to(device).eval())
        blocks.append(models.vgg16(pretrained=True).features[9:16].to(device).eval())
        blocks.append(models.vgg16(pretrained=True).features[16:23].to(device).eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.device=device

    def forward(self, input, target):
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss = loss + torch.nn.MSELoss()(x, y)
        return loss


def PSNR(Img_pred, Img_true):
    return 10 * torch.log10(torch.max(Img_pred)**2 / torch.nn.MSELoss()(Img_pred,Img_true))

def SSIM(Img_pred, Img_true):
    L = torch.max(Img_true) - torch.min(Img_true)
    k1 = 0.01
    k2 = 0.03
    c1 = (k1*L)**2
    c2 = (k2*L)**2
    mu1 = torch.mean(Img_pred)
    mu2 = torch.mean(Img_true)
    sig1 = torch.var(Img_pred)
    sig2 = torch.var(Img_true)
    sig12 = torch.mean((Img_pred - mu1) * (Img_true - mu2))
    return (2*mu1*mu2 + c1)*(2*sig12 + c2)/(mu1**2 + mu2**2 + c1) / (sig1 + sig2 + c2)
