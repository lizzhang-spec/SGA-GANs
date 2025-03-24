import argparse
import os
import numpy as np
import math
import sys
import datetime
import torchvision

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from pytorch_fid.fid_score import calculate_fid_given_paths

from SGA.population_init.population_init import population_init
from SGA.selection.selection import selection
from SGA.crossover.crossover import crossover
from SGA.mutation.mutation import mutation

import torch.nn as nn
import torch.nn.functional as F
import torch

device = 'cuda:0'

def text_create(name):
    desktop_path = "log"
    full_path = desktop_path + name + '.txt'
    file = open(full_path, 'w')

filename = 'began_search_SGA_log'
text_create(filename)
output = sys.stdout
outputfile = open("log" + filename + '.txt', 'w')
sys.stdout = outputfile


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")

parser.add_argument("--dataset", default='../data/cifar-10-batches-py/val', help='name of dataset')

parser.add_argument("--lr_adam", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--lr_sgd", type=float, default=0.01, help="sgd: learning rate")
parser.add_argument("--lr_rms", type=float, default=0.001, help="rmsprop: learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="sgd: momentum")

parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="number of image channels")
parser.add_argument("--save_ckpt_every", type=int, default=20, help="save ckpt every x epochs")
parser.add_argument('--compute_fid', action='store_true', default=True, help='whether or not compute FID')
parser.add_argument('--max_generation', type=int, default=20, help="Maximum number of population iterations")
parser.add_argument('--p_num', type=int, default=10, help="population size")
parser.add_argument('--s1', type=int, default=0.7, help="prob for crossover")
parser.add_argument('--s2', type=int, default=0.1, help="prob for mutation")
parser.add_argument('--opti_num', type=int, default=3, help="Total number of optimisers available")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True


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



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Loss function
adversarial_loss = torch.nn.BCELoss()
max_generation = opt.max_generation
p_num = opt.p_num
opti_num = opt.opti_num

current_fitness_base = range(p_num)
current_fitness = np.asarray(current_fitness_base, dtype=np.float32)

def calculate_fitness(input, fitness_id):

    # Initialize generator and discriminator
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    L1_inited = list(generator.l1.parameters())
    L2_inited = list(generator.conv_blocks_1.parameters())
    L3_inited = list(generator.conv_blocks_2.parameters())
    L4_inited = list(generator.conv_blocks_3.parameters())
    L5_inited = list(discriminator.model1.parameters())
    L6_inited = list(discriminator.model2.parameters())
    L7_inited = list(discriminator.model3.parameters())
    L8_inited = list(discriminator.model4.parameters())

    L1_inited_path = "./log/SGA/pth/first_time_evaluation/p%d/l1.pth" % fitness_id
    L2_inited_path = "./log/SGA/pth/first_time_evaluation/p%d/l2.pth" % fitness_id
    L3_inited_path = "./log/SGA/pth/first_time_evaluation/p%d/l3.pth" % fitness_id
    L4_inited_path = "./log/SGA/pth/first_time_evaluation/p%d/l4.pth" % fitness_id
    L5_inited_path = "./log/SGA/pth/first_time_evaluation/p%d/l5.pth" % fitness_id
    L6_inited_path = "./log/SGA/pth/first_time_evaluation/p%d/l6.pth" % fitness_id
    L7_inited_path = "./log/SGA/pth/first_time_evaluation/p%d/l7.pth" % fitness_id
    L8_inited_path = "./log/SGA/pth/first_time_evaluation/p%d/l8.pth" % fitness_id

    torch.save(L1_inited, L1_inited_path)
    torch.save(L2_inited, L2_inited_path)
    torch.save(L3_inited, L3_inited_path)
    torch.save(L4_inited, L4_inited_path)
    torch.save(L5_inited, L5_inited_path)
    torch.save(L6_inited, L6_inited_path)
    torch.save(L7_inited, L7_inited_path)
    torch.save(L8_inited, L8_inited_path)


    L11_inited = torch.load(L1_inited_path)
    L22_inited = torch.load(L2_inited_path)
    L33_inited = torch.load(L3_inited_path)
    L44_inited = torch.load(L4_inited_path)
    L55_inited = torch.load(L5_inited_path)
    L66_inited = torch.load(L6_inited_path)
    L77_inited = torch.load(L7_inited_path)
    L88_inited = torch.load(L8_inited_path)

    optimizer = [torch.optim.Adam(L11_inited, lr=opt.lr_adam, betas=(opt.b1, opt.b2)),
                 torch.optim.RMSprop(L11_inited, lr=opt.lr_rms, momentum=opt.momentum),
                 torch.optim.SGD(L11_inited, lr=opt.lr_sgd, momentum=opt.momentum),
                 torch.optim.Adam(L22_inited, lr=opt.lr_adam, betas=(opt.b1, opt.b2)),
                 torch.optim.RMSprop(L22_inited, lr=opt.lr_rms, momentum=opt.momentum),
                 torch.optim.SGD(L22_inited, lr=opt.lr_sgd, momentum=opt.momentum),
                 torch.optim.Adam(L33_inited, lr=opt.lr_adam, betas=(opt.b1, opt.b2)),
                 torch.optim.RMSprop(L33_inited, lr=opt.lr_rms, momentum=opt.momentum),
                 torch.optim.SGD(L33_inited, lr=opt.lr_sgd, momentum=opt.momentum),
                 torch.optim.Adam(L44_inited, lr=opt.lr_adam, betas=(opt.b1, opt.b2)),
                 torch.optim.RMSprop(L44_inited, lr=opt.lr_rms, momentum=opt.momentum),
                 torch.optim.SGD(L44_inited, lr=opt.lr_sgd, momentum=opt.momentum),
                 torch.optim.Adam(L55_inited, lr=opt.lr_adam, betas=(opt.b1, opt.b2)),
                 torch.optim.RMSprop(L55_inited, lr=opt.lr_rms, momentum=opt.momentum),
                 torch.optim.SGD(L55_inited, lr=opt.lr_sgd, momentum=opt.momentum),
                 torch.optim.Adam(L66_inited, lr=opt.lr_adam, betas=(opt.b1, opt.b2)),
                 torch.optim.RMSprop(L66_inited, lr=opt.lr_rms, momentum=opt.momentum),
                 torch.optim.SGD(L66_inited, lr=opt.lr_sgd, momentum=opt.momentum),
                 torch.optim.Adam(L77_inited, lr=opt.lr_adam, betas=(opt.b1, opt.b2)),
                 torch.optim.RMSprop(L77_inited, lr=opt.lr_rms, momentum=opt.momentum),
                 torch.optim.SGD(L77_inited, lr=opt.lr_sgd, momentum=opt.momentum),
                 torch.optim.Adam(L88_inited, lr=opt.lr_adam, betas=(opt.b1, opt.b2)),
                 torch.optim.RMSprop(L88_inited, lr=opt.lr_rms, momentum=opt.momentum),
                 torch.optim.SGD(L88_inited, lr=opt.lr_sgd, momentum=opt.momentum),
                 ]

    selected_optimizer_index = []
    selected_optimizer = []

    # select optimizers according to the input
    for i in range(8):
        start_id = opti_num * i
        end_id = opti_num * i + opti_num
        selected_optimizer_index.append(np.argmax(input[start_id: end_id]))
        selected_optimizer.append(optimizer[start_id: end_id][np.argmax(input[start_id: end_id])])
    print('selected_optimizer_index=', selected_optimizer_index)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(opt.img_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = torchvision.datasets.ImageFolder(opt.dataset, transform=transforms)

    dataloader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=True,
                                              drop_last=True
                                              )

    Tensor = torch.cuda.FloatTensor

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

            for idx3 in range(0, 4):
                selected_optimizer[idx3].zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = torch.mean(torch.abs(discriminator(gen_imgs) - gen_imgs))

            g_loss.backward()
            for idx4 in range(0, 4):
                selected_optimizer[idx4].step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            for idx1 in range(4, 8):
                selected_optimizer[idx1].zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            d_real = discriminator(real_imgs)
            d_fake = discriminator(gen_imgs.detach())

            d_loss_real = torch.mean(torch.abs(d_real - real_imgs))
            d_loss_fake = torch.mean(torch.abs(d_fake - gen_imgs.detach()))
            d_loss = d_loss_real - k * d_loss_fake

            d_loss.backward()
            for idx2 in range(4, 8):
                selected_optimizer[idx2].step()

            # ----------------
            # Update weights
            # ----------------

            diff = torch.mean(gamma * d_loss_real - d_loss_fake)

            # Update weight term for fake samples
            k = k + lambda_k * diff.item()
            k = min(max(k, 0), 1)  # Constraint to interval [0, 1]

            # Update convergence metric
            M = (d_loss_real + torch.abs(diff)).item()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] -- M: %f, k: %f"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), M, k)
            )

            batches_done = epoch * len(dataloader) + i
            save_image(gen_imgs, "./log/SGA/output_images/first_time_evaluation/%d.png" % batches_done, normalize=True)

        L1_first_cal = generator.l1.state_dict()
        L2_first_cal = generator.conv_blocks_1.state_dict()
        L3_first_cal = generator.conv_blocks_2.state_dict()
        L4_first_cal = generator.conv_blocks_3.state_dict()
        L5_first_cal = discriminator.model1.state_dict()
        L6_first_cal = discriminator.model2.state_dict()
        L7_first_cal = discriminator.model3.state_dict()
        L8_first_cal = discriminator.model4.state_dict()

        L1_first_cal_path = "./log/SGA/pth/first_time_evaluation/p%d/epoch%d/l1.pth" % (fitness_id, epoch)
        L2_first_cal_path = "./log/SGA/pth/first_time_evaluation/p%d/epoch%d/l2.pth" % (fitness_id, epoch)
        L3_first_cal_path = "./log/SGA/pth/first_time_evaluation/p%d/epoch%d/l3.pth" % (fitness_id, epoch)
        L4_first_cal_path = "./log/SGA/pth/first_time_evaluation/p%d/epoch%d/l4.pth" % (fitness_id, epoch)
        L5_first_cal_path = "./log/SGA/pth/first_time_evaluation/p%d/epoch%d/l5.pth" % (fitness_id, epoch)
        L6_first_cal_path = "./log/SGA/pth/first_time_evaluation/p%d/epoch%d/l6.pth" % (fitness_id, epoch)
        L7_first_cal_path = "./log/SGA/pth/first_time_evaluation/p%d/epoch%d/l7.pth" % (fitness_id, epoch)
        L8_first_cal_path = "./log/SGA/pth/first_time_evaluation/p%d/epoch%d/l8.pth" % (fitness_id, epoch)

        torch.save(L1_first_cal, L1_first_cal_path)
        torch.save(L2_first_cal, L2_first_cal_path)
        torch.save(L3_first_cal, L3_first_cal_path)
        torch.save(L4_first_cal, L4_first_cal_path)
        torch.save(L5_first_cal, L5_first_cal_path)
        torch.save(L6_first_cal, L6_first_cal_path)
        torch.save(L7_first_cal, L7_first_cal_path)
        torch.save(L8_first_cal, L8_first_cal_path)

        if opt.compute_fid:
            torch.manual_seed(42)

            to_range_0_1 = lambda x: (x + 1.) / 2.

            netG = Generator().to(device)

            L11_first_cal = torch.load(L1_first_cal_path)
            L22_first_cal = torch.load(L2_first_cal_path)
            L33_first_cal = torch.load(L3_first_cal_path)
            L44_first_cal = torch.load(L4_first_cal_path)

            netG.l1.load_state_dict(L11_first_cal)
            netG.conv_blocks_1.load_state_dict(L22_first_cal)
            netG.conv_blocks_2.load_state_dict(L33_first_cal)
            netG.conv_blocks_3.load_state_dict(L44_first_cal)

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
                                   "./log/SGA/output_images_for_fid/first_time_evaluation/epoch_{}/{}.png".format(epoch,
                                                                                                                  index),
                                   normalize=True)

            fake_img_dir = "./log/SGA/output_images_for_fid/first_time_evaluation/epoch_{}".format(epoch)
            real_imgs_folder = "../data/val_cifar10_for_fid"

            paths = [fake_img_dir, real_imgs_folder]

            kwargs = {'batch_size': 100, 'device': device, 'dims': 2048}
            fid = calculate_fid_given_paths(paths=paths, **kwargs)
            print('FID_{} = {}'.format(epoch, fid))

            current_epoch_fid = fid

        fitness = d_loss.item() + 1.0 / g_loss.item() + 1.0 / current_epoch_fid
        print(
            "[Epoch %d/%d] [fitness: %f] [D loss: %f] [G loss: %f] -- M: %f, k: %f [current_epoch_fid: %f]"
            % (epoch, opt.n_epochs, fitness, d_loss.item(), g_loss.item(), M, k, current_epoch_fid)
        )

        current_fitness[fitness_id] = fitness.item()

def calculate_fitness_for_evolution(input, fitness_id, iteration):

    # Initialize generator and discriminator
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    L1_inited = list(generator.l1.parameters())
    L2_inited = list(generator.conv_blocks_1.parameters())
    L3_inited = list(generator.conv_blocks_2.parameters())
    L4_inited = list(generator.conv_blocks_3.parameters())
    L5_inited = list(discriminator.model1.parameters())
    L6_inited = list(discriminator.model2.parameters())
    L7_inited = list(discriminator.model3.parameters())
    L8_inited = list(discriminator.model4.parameters())

    L1_inited_path = "./log/SGA/pth/evolutionary_evaluation/iter_%d/p%d/l1.pth" % (iteration, fitness_id)
    L2_inited_path = "./log/SGA/pth/evolutionary_evaluation/iter_%d/p%d/l2.pth" % (iteration, fitness_id)
    L3_inited_path = "./log/SGA/pth/evolutionary_evaluation/iter_%d/p%d/l3.pth" % (iteration, fitness_id)
    L4_inited_path = "./log/SGA/pth/evolutionary_evaluation/iter_%d/p%d/l4.pth" % (iteration, fitness_id)
    L5_inited_path = "./log/SGA/pth/evolutionary_evaluation/iter_%d/p%d/l5.pth" % (iteration, fitness_id)
    L6_inited_path = "./log/SGA/pth/evolutionary_evaluation/iter_%d/p%d/l6.pth" % (iteration, fitness_id)
    L7_inited_path = "./log/SGA/pth/evolutionary_evaluation/iter_%d/p%d/l7.pth" % (iteration, fitness_id)
    L8_inited_path = "./log/SGA/pth/evolutionary_evaluation/iter_%d/p%d/l8.pth" % (iteration, fitness_id)

    torch.save(L1_inited, L1_inited_path)
    torch.save(L2_inited, L2_inited_path)
    torch.save(L3_inited, L3_inited_path)
    torch.save(L4_inited, L4_inited_path)
    torch.save(L5_inited, L5_inited_path)
    torch.save(L6_inited, L6_inited_path)
    torch.save(L7_inited, L7_inited_path)
    torch.save(L8_inited, L8_inited_path)


    L11_inited = torch.load(L1_inited_path)
    L22_inited = torch.load(L2_inited_path)
    L33_inited = torch.load(L3_inited_path)
    L44_inited = torch.load(L4_inited_path)
    L55_inited = torch.load(L5_inited_path)
    L66_inited = torch.load(L6_inited_path)
    L77_inited = torch.load(L7_inited_path)
    L88_inited = torch.load(L8_inited_path)

    optimizer = [torch.optim.Adam(L11_inited, lr=opt.lr_adam, betas=(opt.b1, opt.b2)),
                 torch.optim.RMSprop(L11_inited, lr=opt.lr_rms, momentum=opt.momentum),
                 torch.optim.SGD(L11_inited, lr=opt.lr_sgd, momentum=opt.momentum),
                 torch.optim.Adam(L22_inited, lr=opt.lr_adam, betas=(opt.b1, opt.b2)),
                 torch.optim.RMSprop(L22_inited, lr=opt.lr_rms, momentum=opt.momentum),
                 torch.optim.SGD(L22_inited, lr=opt.lr_sgd, momentum=opt.momentum),
                 torch.optim.Adam(L33_inited, lr=opt.lr_adam, betas=(opt.b1, opt.b2)),
                 torch.optim.RMSprop(L33_inited, lr=opt.lr_rms, momentum=opt.momentum),
                 torch.optim.SGD(L33_inited, lr=opt.lr_sgd, momentum=opt.momentum),
                 torch.optim.Adam(L44_inited, lr=opt.lr_adam, betas=(opt.b1, opt.b2)),
                 torch.optim.RMSprop(L44_inited, lr=opt.lr_rms, momentum=opt.momentum),
                 torch.optim.SGD(L44_inited, lr=opt.lr_sgd, momentum=opt.momentum),
                 torch.optim.Adam(L55_inited, lr=opt.lr_adam, betas=(opt.b1, opt.b2)),
                 torch.optim.RMSprop(L55_inited, lr=opt.lr_rms, momentum=opt.momentum),
                 torch.optim.SGD(L55_inited, lr=opt.lr_sgd, momentum=opt.momentum),
                 torch.optim.Adam(L66_inited, lr=opt.lr_adam, betas=(opt.b1, opt.b2)),
                 torch.optim.RMSprop(L66_inited, lr=opt.lr_rms, momentum=opt.momentum),
                 torch.optim.SGD(L66_inited, lr=opt.lr_sgd, momentum=opt.momentum),
                 torch.optim.Adam(L77_inited, lr=opt.lr_adam, betas=(opt.b1, opt.b2)),
                 torch.optim.RMSprop(L77_inited, lr=opt.lr_rms, momentum=opt.momentum),
                 torch.optim.SGD(L77_inited, lr=opt.lr_sgd, momentum=opt.momentum),
                 torch.optim.Adam(L88_inited, lr=opt.lr_adam, betas=(opt.b1, opt.b2)),
                 torch.optim.RMSprop(L88_inited, lr=opt.lr_rms, momentum=opt.momentum),
                 torch.optim.SGD(L88_inited, lr=opt.lr_sgd, momentum=opt.momentum),
                 ]

    selected_optimizer_index = []
    selected_optimizer = []

    # select optimizers according to the input
    for i in range(8):
        start_id = opti_num * i
        end_id = opti_num * i + opti_num
        selected_optimizer_index.append(np.argmax(input[start_id: end_id]))
        selected_optimizer.append(optimizer[start_id: end_id][np.argmax(input[start_id: end_id])])
    print('selected_optimizer_index=', selected_optimizer_index)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(opt.img_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = torchvision.datasets.ImageFolder(opt.dataset, transform=transforms)

    dataloader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=True,
                                              drop_last=True
                                              )

    Tensor = torch.cuda.FloatTensor

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

            for idx3 in range(0, 4):
                selected_optimizer[idx3].zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = torch.mean(torch.abs(discriminator(gen_imgs) - gen_imgs))

            g_loss.backward()
            for idx4 in range(0, 4):
                selected_optimizer[idx4].step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            for idx1 in range(4, 8):
                selected_optimizer[idx1].zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            d_real = discriminator(real_imgs)
            d_fake = discriminator(gen_imgs.detach())

            d_loss_real = torch.mean(torch.abs(d_real - real_imgs))
            d_loss_fake = torch.mean(torch.abs(d_fake - gen_imgs.detach()))
            d_loss = d_loss_real - k * d_loss_fake

            d_loss.backward()
            for idx2 in range(4, 8):
                selected_optimizer[idx2].step()

            # ----------------
            # Update weights
            # ----------------

            diff = torch.mean(gamma * d_loss_real - d_loss_fake)

            # Update weight term for fake samples
            k = k + lambda_k * diff.item()
            k = min(max(k, 0), 1)  # Constraint to interval [0, 1]

            # Update convergence metric
            M = (d_loss_real + torch.abs(diff)).item()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] -- M: %f, k: %f"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), M, k)
            )

            batches_done = epoch * len(dataloader) + i
            save_image(gen_imgs, "./log/SGA/output_images/evolutionary_evaluation/iter_%d/%d.png" % (iteration, batches_done), normalize=True)

        L1_first_cal = generator.l1.state_dict()
        L2_first_cal = generator.conv_blocks_1.state_dict()
        L3_first_cal = generator.conv_blocks_2.state_dict()
        L4_first_cal = generator.conv_blocks_3.state_dict()
        L5_first_cal = discriminator.model1.state_dict()
        L6_first_cal = discriminator.model2.state_dict()
        L7_first_cal = discriminator.model3.state_dict()
        L8_first_cal = discriminator.model4.state_dict()

        L1_first_cal_path = "./log/SGA/pth/evolutionary_evaluation/iter_%d/p%d/epoch%d/l1.pth" % (iteration, fitness_id, epoch)
        L2_first_cal_path = "./log/SGA/pth/evolutionary_evaluation/iter_%d/p%d/epoch%d/l2.pth" % (iteration, fitness_id, epoch)
        L3_first_cal_path = "./log/SGA/pth/evolutionary_evaluation/iter_%d/p%d/epoch%d/l3.pth" % (iteration, fitness_id, epoch)
        L4_first_cal_path = "./log/SGA/pth/evolutionary_evaluation/iter_%d/p%d/epoch%d/l4.pth" % (iteration, fitness_id, epoch)
        L5_first_cal_path = "./log/SGA/pth/evolutionary_evaluation/iter_%d/p%d/epoch%d/l5.pth" % (iteration, fitness_id, epoch)
        L6_first_cal_path = "./log/SGA/pth/evolutionary_evaluation/iter_%d/p%d/epoch%d/l6.pth" % (iteration, fitness_id, epoch)
        L7_first_cal_path = "./log/SGA/pth/evolutionary_evaluation/iter_%d/p%d/epoch%d/l7.pth" % (iteration, fitness_id, epoch)
        L8_first_cal_path = "./log/SGA/pth/evolutionary_evaluation/iter_%d/p%d/epoch%d/l8.pth" % (iteration, fitness_id, epoch)

        torch.save(L1_first_cal, L1_first_cal_path)
        torch.save(L2_first_cal, L2_first_cal_path)
        torch.save(L3_first_cal, L3_first_cal_path)
        torch.save(L4_first_cal, L4_first_cal_path)
        torch.save(L5_first_cal, L5_first_cal_path)
        torch.save(L6_first_cal, L6_first_cal_path)
        torch.save(L7_first_cal, L7_first_cal_path)
        torch.save(L8_first_cal, L8_first_cal_path)

        if opt.compute_fid:
            torch.manual_seed(42)

            to_range_0_1 = lambda x: (x + 1.) / 2.

            netG = Generator().to(device)

            L11_first_cal = torch.load(L1_first_cal_path)
            L22_first_cal = torch.load(L2_first_cal_path)
            L33_first_cal = torch.load(L3_first_cal_path)
            L44_first_cal = torch.load(L4_first_cal_path)

            netG.l1.load_state_dict(L11_first_cal)
            netG.conv_blocks_1.load_state_dict(L22_first_cal)
            netG.conv_blocks_2.load_state_dict(L33_first_cal)
            netG.conv_blocks_3.load_state_dict(L44_first_cal)

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
                                   "./log/SGA/output_images_for_fid/evolutionary_evaluation/iter_{}/epoch_{}/{}.png".format(iteration, epoch, index), normalize=True)

            fake_img_dir = "./log/SGA/output_images_for_fid/evolutionary_evaluation/iter_{}/epoch_{}".format(iteration,epoch)
            real_imgs_folder = "../data/val_cifar10_for_fid"

            paths = [fake_img_dir, real_imgs_folder]

            kwargs = {'batch_size': 100, 'device': device, 'dims': 2048}
            fid = calculate_fid_given_paths(paths=paths, **kwargs)
            print('FID_{} = {}'.format(epoch, fid))

            current_epoch_fid = fid

        fitness = d_loss.item() + 1.0/g_loss.item() + 1.0/current_epoch_fid
        print(
            "[Epoch %d/%d] [fitness: %f] [D loss: %f] [G loss: %f] -- M: %f, k: %f [current_epoch_fid: %f]"
            % (epoch, opt.n_epochs, fitness, d_loss.item(), g_loss.item(), M, k, current_epoch_fid)
        )

        current_fitness[fitness_id] = fitness.item()


print("000000000000000000000000000000000000 A new start training 000000000000000000000000000000000000")

if os.path.exists('./log/SGA') == False:
    os.makedirs('./log/SGA')

s1 = opt.s1  # prob for crossover
s2 = opt.s2  # prob for mutation
min_init_Val=(-1,) * 24
max_init_Val=(1,) * 24
eta_c=1
eta_m=1

# Initialize population
P = []
population_init(P, p_num, 24, min_init_Val, max_init_Val)
print('Initialized population=', P)

# xReal=[]

starttime = datetime.datetime.now()
if __name__ == '__main__':

    for i in range(p_num):
        calculate_fitness(P[i], i)

    endtime = datetime.datetime.now()
    print("Time consumption for the first time:", (endtime - starttime).seconds)

    P_best = P[np.argmax(current_fitness)]
    np.savetxt('./log/SGA/txt/best_optimizers_config_for_first_time.txt', P_best, delimiter=',')

    best_fitness = max(current_fitness)
    ave_fitness = np.mean(current_fitness)
    print('The best fitness for the first time is: %4f' % (best_fitness))
    print('The ave fitness for the first time is: %4f' % (ave_fitness))


    print('000000000000000000000000000000000000 Iteration Start 000000000000000000000000000000000000')


    for j in range(max_generation):

        selection(P, current_fitness)
        crossover(P, s1, 24, min_init_Val, max_init_Val, eta_c)
        mutation(P, s2, 24, min_init_Val, max_init_Val, eta_m)

        for i in range(p_num):
            calculate_fitness_for_evolution(P[i], i, j)

        P_best = P[np.argmax(current_fitness)]
        np.savetxt('./log/SGA/txt/best_optimizers_generation_%d_th.txt' % j, P_best, delimiter=',')

        best_fitness = max(current_fitness)
        ave_fitness = np.mean(current_fitness)
        print('The best fitness of the %d-th generation is: %4f' % (j, best_fitness))
        print('The ave fitness of the %d-th generation is: %4f' % (j, ave_fitness))







outputfile.close()
