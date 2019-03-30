import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from tensorboardX import SummaryWriter


class JusticeGAN():
	def __init__(self, opt):
		self.opt = opt
		
		# === model === #
		# Criminal (犯罪者)
		self.C = nn.Sequential(
			nn.Linear(opt.nz, opt.ncf, bias=False),
			nn.BatchNorm1d(opt.ncf),
			nn.ReLU(True),
			nn.Linear(opt.ncf, opt.ncf, bias=False),
			nn.BatchNorm1d(opt.ncf),
			nn.ReLU(True),
			nn.Linear(opt.ncf, opt.ncf, bias=False),
			nn.BatchNorm1d(opt.ncf),
			nn.ReLU(True),
			nn.Linear(opt.ncf, opt.ncf, bias=False),
			nn.BatchNorm1d(opt.ncf),
			nn.ReLU(True),
			nn.Linear(opt.ncf, 2)
		).to(opt.device)

		# Offer (状況証拠)
		self.O = nn.Sequential(
			nn.Linear(2, opt.njf, bias=False),
			nn.BatchNorm1d(opt.njf),
			nn.ReLU(True),
			nn.Linear(opt.njf, opt.njf, bias=False),
		).to(opt.device)

		# Lawyer (弁護士)
		self.L = nn.Sequential(
			nn.Linear(opt.njf, opt.njf, bias=False)
		).to(opt.device)

		# Judge (裁判官)
		self.J = nn.Sequential(
			nn.BatchNorm1d(opt.njf),
			nn.ReLU(True),
			nn.Linear(opt.njf, opt.njf, bias=False),
			nn.BatchNorm1d(opt.njf),
			nn.ReLU(True),
			nn.Linear(opt.njf, 1),
			nn.Sigmoid()
		).to(opt.device)

		# === criterion === #
		self.criterion = nn.BCELoss()

		# === optimizers === #
		self.init_optimizers()

	def train(self, perturbation=True):
		self.C.train()
		self.O.train()
		self.L.train()
		self.J.train()

		C_running_loss = 0.0
		O_running_loss = 0.0
		L_running_loss = 0.0
		J_running_loss = 0.0

		nitrs = 0

		for itr, (x) in enumerate(self.opt.loader):
			nitrs += 1

			# --- label --- #
			real_label = torch.full((self.opt.batch_size, 1), 1, device=self.opt.device)
			fake_label = torch.full((self.opt.batch_size, 1), 0, device=self.opt.device)

			# --- Update J network --- #
			self.J_optimizer.zero_grad()

			x_real = x[0].to(self.opt.device)
			v_real = self.O(x_real)
			if perturbation:
				l_real = self.L(v_real)
			else:
				l_real = torch.zeros_like(v_real)
			j_real = self.J(v_real + l_real)

			z = torch.randn((self.opt.batch_size, self.opt.nz), device=self.opt.device)
			x_fake = self.C(z)
			v_fake = self.O(x_fake)
			if perturbation:
				l_fake = self.L(v_fake)
			else:
				l_fake = torch.zeros_like(v_fake)
			j_fake = self.J(v_fake + l_fake)

			loss_real = self.criterion(j_real, real_label)
			loss_fake = self.criterion(j_fake, fake_label)
			J_loss = loss_real + loss_fake
			J_loss.backward()
			self.J_optimizer.step()

			J_running_loss += J_loss.item()

			# --- Update O network --- #
			self.O_optimizer.zero_grad()

			x_real = x[0].to(self.opt.device)
			v_real = self.O(x_real)
			if perturbation:
				l_real = self.L(v_real)
			else:
				l_real = torch.zeros_like(v_real)
			j_real = self.J(v_real + l_real)

			z = torch.randn((self.opt.batch_size, self.opt.nz), device=self.opt.device)
			x_fake = self.C(z)
			v_fake = self.O(x_fake)
			if perturbation:
				l_fake = self.L(v_fake)
			else:
				l_fake = torch.zeros_like(v_fake)
			j_fake = self.J(v_fake + l_fake)

			loss_real = self.criterion(j_real, real_label)
			loss_fake = self.criterion(j_fake, fake_label)

			O_loss = loss_real + loss_fake
			O_loss.backward()
			self.O_optimizer.step()

			O_running_loss += O_loss.item()

			if (itr + 1) % self.opt.njitrs:
				if perturbation:
					# --- Update P & L network --- #
					self.L_optimizer.zero_grad()

					x_real = x[0].to(self.opt.device)
					v_real = self.O(x_real)
					if perturbation:
						l_real = self.L(v_real)
					else:
						l_real = torch.zeros_like(v_real)
					j_real = self.J(v_real + l_real)

					z = torch.randn((self.opt.batch_size, self.opt.nz), device=self.opt.device)
					x_fake = self.C(z)
					v_fake = self.O(x_fake)
					if perturbation:
						l_fake = self.L(v_fake)
					else:
						l_fake = torch.zeros_like(v_fake)
					j_fake = self.J(v_fake + l_fake)

					L_loss = torch.mean(j_real) + self.opt.gamma * (torch.mean(torch.abs(l_real)) + torch.mean(torch.abs(l_fake)))
					# L_loss = torch.mean(F.relu(j_fake - 0.5)) + self.opt.gamma * torch.mean(torch.abs(l_real)) + self.opt.gamma * torch.mean(torch.abs(l_fake))
					L_loss.backward(retain_graph=True)
					self.L_optimizer.step()

					L_running_loss += L_loss.item()

				# --- Update C network --- #
				self.C_optimizer.zero_grad()

				z = torch.randn((self.opt.batch_size, self.opt.nz), device=self.opt.device)
				x_fake = self.C(z)
				v_fake = self.O(x_fake)
				if perturbation:
					l_fake = self.L(v_fake)
				else:
					l_fake = torch.zeros_like(v_fake)
				j_fake = self.J(v_fake + l_fake)

				C_loss = self.criterion(j_fake, real_label)
				C_loss.backward()
				self.C_optimizer.step()

				C_running_loss += C_loss.item()

		return C_running_loss / nitrs, O_running_loss / nitrs, L_running_loss / nitrs, J_running_loss / nitrs

	def generate(self):
		self.C.eval()
		return self.C(torch.randn((self.opt.batch_size, self.opt.nz), device=self.opt.device))

	def test(self):
		generated_samples = []
		for i in range(self.opt.num_test_samples // self.opt.batch_size):
			generated_samples.append(self.generate())
		return torch.cat(generated_samples).cpu().detach().numpy()

	def init_optimizers(self):
		# === optimizers === #
		self.C_optimizer = optim.Adam(self.C.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
		self.O_optimizer = optim.Adam(self.O.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
		self.L_optimizer = optim.Adam(self.L.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
		self.J_optimizer = optim.Adam(self.J.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--nz', type=int, default=128, help='latent variable size')
	parser.add_argument('--ncf', type=int, default=64, help='feature size of criminal')
	parser.add_argument('--njf', type=int, default=64, help='feature size of judge')
	parser.add_argument('--njitrs', type=int, default=5, help='number of iterations for updating juddge')
	parser.add_argument('--gamma', type=float, default=1.0, help='coefficient of perturbation loss')

	parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
	parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
	parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
	parser.add_argument('--batch_size', type=int, default=8000, help='batch size')
	parser.add_argument('--num_epochs', type=int, default=500, help='number of epochs')
	parser.add_argument('--deactivation_epoch', type=int, default=250, help='epoch to deactivate L')
	parser.add_argument('--use_gpu', action='store_true', default=False, help='GPU mode')

	parser.add_argument('--log_dir', type=str, default='logs', help='log directory')
	parser.add_argument('--num_test_samples', type=int, default=80000, help='number of samples in test')
	parser.add_argument('--checkpoint', type=int, default=10, help='checkpoint epoch')
	opt = parser.parse_args()

	opt.device = torch.device("cuda:0" if opt.use_gpu else "cpu")
	writer = SummaryWriter(os.path.join(opt.log_dir, 'runs'))

	# dataset
	tensor_data, dataset = generate_dataset()
	np_data = tensor_data.numpy()
	opt.loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

	model = JusticeGAN(opt)

	for epoch in range(1, opt.num_epochs+1):
		if epoch < opt.deactivation_epoch:
			C_loss, O_loss, L_loss, J_loss = model.train(perturbation=True)
		else:
			C_loss, O_loss, L_loss, J_loss = model.train(perturbation=False)

		if epoch % opt.checkpoint == 0:
			generated_samples = model.test()

			fig = plt.figure()		
			plt.scatter(np_data[:, 0], np_data[:, 1], s=1, c="red")
			plt.scatter(generated_samples[:, 0], generated_samples[:, 1], s=1, c="blue")
			plt.axes().set_aspect('equal', 'datalim')
			plt.savefig(os.path.join(opt.log_dir, 'epoch{:04d}.png'.format(epoch)), dpi=100)
			plt.close(fig)

		print('[epoch {:4d}] C: {:.4f}, O: {:.4f}, L: {:.4f}, J: {:.4f}'.format(epoch, C_loss, O_loss, L_loss, J_loss))
		writer.add_scalars('Loss', {'C': C_loss, 'O': O_loss, "L": L_loss, "J": J_loss}, global_step=epoch)



def generate_dataset(radius=1.0, npoints=8, std=0.1, num_samples=10000):
	means = []
	for p in range(npoints):
		theta = p * 2.0 * np.pi / npoints
		means.append(torch.tensor([radius * float(np.cos(theta)), radius * float(np.sin(theta))]))
	data = []
	for i in range(npoints):
		data.append(means[i] + std * torch.randn(num_samples, 2))
	tensor_data = torch.cat(data)
	dataset = TensorDataset(tensor_data)
	return tensor_data, dataset


if __name__ == '__main__':
	main()