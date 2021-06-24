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

        g_loss_val = 0
        d_loss_val = 0
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
            d_loss_val += d_loss.cpu().item()
            d_loss.backward(retain_graph=True)
            d_optim.step()

            # train G
            g_optim.zero_grad()
            fake_logits = dis(fake_img)
            g_loss = self.d_loss_func(fake_logits, real_label) + self.g_loss_func(fake_img, batch_label)
            g_loss_val += g_loss.cpu().item()
            g_loss.backward(retain_graph=False)
            g_optim.step()

        return g_loss_val, d_loss_val

    def test_step(self, gen, data_loader):
        gen.eval()
        
        temp_dir = os.path.join(self.output_dir, str(self.epoch))
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)

        with torch.no_grad():
            for batch_img, batch_angle, batch_label in enumerate(dataloader):
                111

            fake_img = gen(noise_label)
            plt.figure(figsize=(10, 10))
            for i in range(10):
                for j in range(10):
                    plt.subplot(10, 10, i * 10 + j + 1)
                    plt.imshow(fake_img[i * 10 + j, 0, :, :].cpu().numpy() * 127.5 + 127.5, cmap='gray')
                    plt.axis(False)
            plt.savefig(utils.path.join(utils.FAKE_MNIST_DIR, str(self.epoch) + '.png'))
            plt.tight_layout()
            plt.close()

    def train(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        else:
            shutil.rmtree(self.output_dir)
            os.mkdir(self.output_dir)
        
        dataset = BoneDataSet()
        # print('dataset: ', len(dataset))
        data_loader = DataLoader(dataset, batch_size = self.batch_size)
        test_loader = DataLoader(dataset, batch_size = 1)

        gen = Generator().to(device)
        dis = Discriminator().to(device)

        d_optim = torch.optim.Adam(dis.parameters(), lr=0.005) # 0.00005
        g_optim = torch.optim.Adam(gen.parameters(), lr=0.05) # 0.00015

        for _ in range(self.epochs):
            g_loss, d_loss = self.train_step(gen, dis, data_loader, g_optim, d_optim)
            print('Epoch {}/{}, g_loss = {:.4f}, d_loss = {:.4f}'.format(self.epoch + 1, self.epochs, g_loss, d_loss))
            # self.test_step(gen, test_loader)
            self.epoch += 1

if __name__ == '__main__':
    gan = CDCGAN(batch_size=2)
    gan.train()
