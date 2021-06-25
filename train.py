# coding: utf-8
import torch
import torchvision
import tqdm
import os
import shutil
import matplotlib.pyplot as plt
from dataset import BoneDataSet
from torch.utils.data import DataLoader
from torch import nn
from model import Discriminator, Generator
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CDCGAN:
    def __init__(self, epochs=50, batch_size=256):
        self.epochs = epochs
        self.epoch = 0
        self.batch_size = batch_size
        self.g_loss_func = nn.MSELoss()
        self.d_loss_func = nn.CrossEntropyLoss()
        self.output_dir = './output'

    def train_step(self, gen, dis, train_loader, g_optim, d_optim):
        gen.train()
        dis.train()

        g_loss_cls = 0
        g_loss_mse = 0
        # g_loss_val = 0
        d_loss_real = 0
        d_loss_fake = 0
        num_of_batch = len(train_loader)
        desc = f'Epoch {self.epoch + 1}/{self.epochs}, Step'
        data_loader_itr = iter(train_loader)

        for _ in tqdm.trange(num_of_batch, desc=desc, total=num_of_batch):
            batch_img, batch_angle, batch_label = next(data_loader_itr)
            noise = torch.randn(train_loader.batch_size, 1, batch_img.shape[1], batch_img.shape[2]).to(device)
            
            batch_img = batch_img.unsqueeze(1).to(device)
            batch_label = batch_label.unsqueeze(1).to(device)
            batch_angle = torch.nn.functional.one_hot(batch_angle, num_classes=7).float().to(device)

            noise_label_img = torch.add(batch_img, noise).to(device)
            fake_img = gen(noise_label_img, batch_angle)

            # train D
            d_optim.zero_grad()
            real_logits = dis(batch_label)            
            fake_logits = dis(fake_img)
            real_label = torch.ones(train_loader.batch_size).long().to(device)
            fake_label = torch.zeros(train_loader.batch_size).long().to(device)
            d_loss = self.d_loss_func(real_logits, real_label) + self.d_loss_func(fake_logits, fake_label)
            # d_loss_val += d_loss.cpu().item()
            d_loss_real += self.d_loss_func(real_logits, real_label).cpu().item()
            d_loss_fake += self.d_loss_func(fake_logits, fake_label).cpu().item()
            d_loss.backward(retain_graph=True)
            d_optim.step()

            # train G
            g_optim.zero_grad()
            fake_logits = dis(fake_img)
            g_loss = self.d_loss_func(fake_logits, real_label) # + self.g_loss_func(fake_img, batch_label)
            # g_loss_val += g_loss.cpu().item()
            g_loss_cls += self.d_loss_func(fake_logits, real_label).cpu().item()
            g_loss_mse += self.g_loss_func(fake_img, batch_label).cpu().item()
            g_loss.backward(retain_graph=False)
            g_optim.step()

        return g_loss_cls, g_loss_mse, d_loss_real, d_loss_fake

    def test_step(self, gen, dataloader):
        gen.eval()     
        test_mse = 0

        with torch.no_grad():
            for batch, (batch_img, batch_angle, batch_label) in enumerate(dataloader):
                batch_img = batch_img.unsqueeze(1).to(device)
                batch_label = batch_label.unsqueeze(1).to(device)

                noise = torch.randn(1, 1, batch_img.shape[1], batch_img.shape[2]).to(device)
                batch_angle = torch.nn.functional.one_hot(batch_angle, num_classes=7).float().to(device)
                noise_label_img = torch.add(batch_img, noise).to(device)
                fake_img = gen(noise_label_img, batch_angle)

                test_mse += self.g_loss_func(fake_img, batch_label).cpu().item()

                temp_dir = os.path.join(self.output_dir, str(self.epoch))
                if not os.path.exists(temp_dir):
                    os.mkdir(temp_dir)

                plt.subplot(1,2,1)
                plt.imshow(fake_img[0, 0, :, :].cpu().numpy() * 127.5 + 127.5, cmap='gray')
                plt.axis(False)
                plt.subplot(1,2,2)
                plt.imshow(batch_label[0, 0, :, :].cpu().numpy() * 255, cmap='gray')
                plt.axis(False)
                plt.savefig(os.path.join(temp_dir, str(batch) + '.png'))
                # plt.tight_layout()
                plt.close()

        print('Test MSE: ', test_mse)

    def train(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        else:
            shutil.rmtree(self.output_dir)
            os.mkdir(self.output_dir)
        
        dataset = BoneDataSet()
        # print('dataset: ', len(dataset))
        data_loader = DataLoader(dataset, batch_size = self.batch_size, shuffle=True)
        test_loader = DataLoader(dataset, batch_size = 1)

        gen = Generator().to(device)
        dis = Discriminator().to(device)

        d_optim = torch.optim.Adam(dis.parameters(), lr=0.00005) # 0.00005
        g_optim = torch.optim.Adam(gen.parameters(), lr=0.005) # 0.00015

        for _ in range(self.epochs):
            g_cls_loss, g_mse_loss, d_real_loss, d_fake_loss = self.train_step(gen, dis, data_loader, g_optim, d_optim)
            print('Epoch {}/{}, g_cls_loss = {:.4f}, g_mse_loss = {:.4f}, d_real_loss = {:.4f}, d_fake_loss = {:.4f}'.format(self.epoch + 1, self.epochs, g_cls_loss, g_mse_loss, d_real_loss, d_fake_loss))
            self.test_step(gen, test_loader)
            print('--------------------------------------------------------------')
            self.epoch += 1

if __name__ == '__main__':
    gan = CDCGAN(batch_size = 4)
    gan.train()
