import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from pytorch_fid.fid_score import calculate_fid_given_paths

import torch.nn as nn
import torch.nn.functional as F
import torch

device = 'cuda:0'

# 创建一个txt文件，文件名为mytxtfile
def text_create(name):
    desktop_path = "log"
    # 新创建的txt文件的存放路径
    full_path = desktop_path + name + '.txt'  # 也可以创建一个.doc的word文档
    file = open(full_path, 'w')

filename = 'began_train_log'
text_create(filename)
output = sys.stdout
outputfile = open("log" + filename + '.txt', 'w')
sys.stdout = outputfile

os.makedirs("../output_images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")

parser.add_argument("--lr_adam", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--lr_sgd", type=float, default=0.01, help="sgd: learning rate")
parser.add_argument("--lr_rms", type=float, default=0.001, help="rmsprop: learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="sgd: momentum")

parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--save_ckpt_every", type=int, default=20, help="save ckpt every x epochs")
parser.add_argument('--compute_fid', action='store_true', default=True, help='whether or not compute FID')

opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks_1 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1))
        self.conv_blocks_2 = nn.Sequential(
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1))
        self.conv_blocks_3 = nn.Sequential(
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        out_1 = self.conv_blocks_1(out)
        out_2 = self.conv_blocks_2(out_1)
        img = self.conv_blocks_3(out_2)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Upsampling
        self.model1 = nn.Sequential(nn.Conv2d(opt.channels, 64, 3, 2, 1), nn.ReLU())
        # Fully-connected layers
        self.down_size = opt.img_size // 2
        down_dim = 64 * (opt.img_size // 2) ** 2
        self.model2 = nn.Sequential(
            nn.Linear(down_dim, 32),
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True))
        self.model3 = nn.Sequential(
            nn.Linear(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        )
        # Upsampling
        self.model4 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, opt.channels, 3, 1, 1))

    def forward(self, img):
        out = self.model1(img)
        out_1 = self.model2(out.view(out.size(0), -1))
        out_2 = self.model3(out_1.view(out_1.size(0), -1))
        out = self.model4(out_2.view(out_2.size(0), 64, self.down_size, self.down_size))
        return out


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
dataloader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
    drop_last=True,
)

# Optimizers
para0 = generator.l1.parameters()
para1 = generator.conv_blocks_1.parameters()
para2 = generator.conv_blocks_2.parameters()
para3 = generator.conv_blocks_3.parameters()
para4 = discriminator.model1.parameters()
para5 = discriminator.model2.parameters()
para6 = discriminator.model3.parameters()
para7 = discriminator.model4.parameters()
# Optimizers
#[0, 2, 0, 0, 2, 1, 1, 0]
optimizer_G_0 = torch.optim.Adam(para0, lr=opt.lr_adam, betas=(opt.b1, opt.b2))
optimizer_G_1 = torch.optim.SGD(para1, lr=opt.lr_sgd, momentum=opt.momentum)
optimizer_G_2 = torch.optim.Adam(para2, lr=opt.lr_adam, betas=(opt.b1, opt.b2))
optimizer_G_3 = torch.optim.Adam(para3, lr=opt.lr_adam, betas=(opt.b1, opt.b2))
optimizer_D_0 = torch.optim.SGD(para4, lr=opt.lr_sgd, momentum=opt.momentum)
optimizer_D_1 = torch.optim.RMSprop(para5, lr=opt.lr_rms, momentum=opt.momentum)
optimizer_D_2 = torch.optim.RMSprop(para6, lr=opt.lr_rms, momentum=opt.momentum)
optimizer_D_3 = torch.optim.Adam(para7, lr=opt.lr_adam, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

# BEGAN hyper parameters
gamma = 0.75
lambda_k = 0.001
k = 0.0

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G_0.zero_grad()
        optimizer_G_1.zero_grad()
        optimizer_G_2.zero_grad()
        optimizer_G_3.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = torch.mean(torch.abs(discriminator(gen_imgs) - gen_imgs))

        g_loss.backward()
        optimizer_G_0.step()
        optimizer_G_1.step()
        optimizer_G_2.step()
        optimizer_G_3.step()


        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D_0.zero_grad()
        optimizer_D_1.zero_grad()
        optimizer_D_2.zero_grad()
        optimizer_D_3.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        d_real = discriminator(real_imgs)
        d_fake = discriminator(gen_imgs.detach())

        d_loss_real = torch.mean(torch.abs(d_real - real_imgs))
        d_loss_fake = torch.mean(torch.abs(d_fake - gen_imgs.detach()))
        d_loss = d_loss_real - k * d_loss_fake

        d_loss.backward()
        optimizer_D_0.step()
        optimizer_D_1.step()
        optimizer_D_2.step()
        optimizer_D_3.step()


        # ----------------
        # Update weights
        # ----------------

        diff = torch.mean(gamma * d_loss_real - d_loss_fake)

        # Update weight term for fake samples
        k = k + lambda_k * diff.item()
        k = min(max(k, 0), 1)  # Constraint to interval [0, 1]

        # Update convergence metric
        M = (d_loss_real + torch.abs(diff)).item()

        # --------------
        # Log Progress
        # --------------

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] -- M: %f, k: %f"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), M, k)
        )

        batches_done = epoch * len(dataloader) + i
        save_image(gen_imgs, "../output_images/began_train/single/%d.png" % batches_done, normalize=True)
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "../output_images/began_train/collage/%d.png" % batches_done, nrow=5,
                       normalize=True)

    pth_path = "../ckpt/began_train"

    if epoch % opt.save_ckpt_every == 0:
        torch.save(generator.state_dict(), os.path.join(pth_path, 'netG_{}.pth'.format(epoch)))

        if opt.compute_fid:
            torch.manual_seed(42)

            to_range_0_1 = lambda x: (x + 1.) / 2.

            netG = Generator().to(device)
            ckpt = torch.load("../ckpt/began_train/netG_{}.pth".format(epoch), map_location=device)

            netG.load_state_dict(ckpt, strict=False)
            netG.eval()

            iters_needed = 50000 // opt.batch_size

            noise = Variable(Tensor(np.random.normal(0, 1, (opt.channels, opt.latent_dim))))

            for i in range(iters_needed):
                with torch.no_grad():
                    generated_images = netG(noise).detach().cpu()
                    generated_images = to_range_0_1(generated_images)
                    for j, x in enumerate(generated_images):
                        index = i * opt.batch_size + j
                        save_image(generated_images[0],
                                   "../fid_output_images/began_train/epoch_{}/{}.png".format(epoch, index),
                                   normalize=True)

            fake_img_dir = "../fid_output_images/began_train/epoch_{}".format(epoch)
            real_imgs_folder = "../data/full_cifar10_for_fid"

            paths = [fake_img_dir, real_imgs_folder]

            kwargs = {'batch_size': 100, 'device': device, 'dims': 2048}
            fid = calculate_fid_given_paths(paths=paths, **kwargs)
            print('FID_{} = {}'.format(epoch, fid))

outputfile.close()
