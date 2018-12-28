import sys

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from tensorboardX import SummaryWriter

# === hyperparameter === #
nz = 16
nh = 8
no = 2
nf = 8

gamma = 1.0
nj = 10

use_gpu = True
device = torch.device("cuda:0" if use_gpu else "cpu")
num_epochs = 100
num_test_samples = 80000
lr = 0.1
batch_size = 8000

writer = SummaryWriter('runs/lr{:.3f}_epoch{:d}'.format(lr, num_epochs))
output_filename = 'result_lr{:.3f}_epoch{:d}.png'.format(lr, num_epochs)

# === dataset === #
radius = 1.0
npoints = 8
means = []
for p in range(npoints):
	theta = p * 2.0 * np.pi / npoints
	means.append(torch.tensor([radius * float(np.cos(theta)), radius * float(np.sin(theta))]))
std = 0.1
nsamples = 10000
data = []
for i in range(npoints):
	data.append(means[i] + std * torch.randn(nsamples, 2))
tensor_data = torch.cat(data)
np_data = tensor_data.numpy()
dataset = TensorDataset(tensor_data)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === model === #
# Criminal (犯罪者)
C = nn.Sequential(
  nn.Linear(nz, nh),
  nn.BatchNorm1d(nh),
  nn.ReLU(True),
  nn.Linear(nh, nh),
  nn.BatchNorm1d(nh),
  nn.ReLU(True),
  nn.Linear(nh, nh),
  nn.BatchNorm1d(nh),
  nn.ReLU(True),
  nn.Linear(nh, nh),
  nn.BatchNorm1d(nh),
  nn.ReLU(True),
  nn.Linear(nh, nh),
  nn.BatchNorm1d(nh),
  nn.ReLU(True),
  nn.Linear(nh, nh),
  nn.BatchNorm1d(nh),
  nn.ReLU(True),
  nn.Linear(nh, no),
  nn.Tanh()
).to(device)

# Offer (状況証拠)
O = nn.Sequential(
  nn.Linear(no, nf),
  nn.BatchNorm1d(nf),
  nn.ReLU(True),
  nn.Linear(nf, nf),
  nn.BatchNorm1d(nf),
  nn.ReLU(True),
  nn.Linear(nf, nf),
  nn.BatchNorm1d(nf),
  nn.ReLU(True),
  nn.Linear(nf, nf),
  nn.BatchNorm1d(nf),
  nn.ReLU(True)
).to(device)

# Lawyer (弁護士)
L = nn.Sequential(
	nn.Linear(nf, nf),
	nn.BatchNorm1d(nf),
	nn.ReLU(True)
).to(device)

# Prosecutor (検察官)
P = nn.Sequential(
	nn.Linear(nf, nf),
	nn.BatchNorm1d(nf),
	nn.ReLU(True)
).to(device)

# Judge (裁判官)
J = nn.Sequential(
	nn.Linear(2 * nf, 1),
	nn.Sigmoid()
).to(device)

# === criterion === #
criterion = nn.BCELoss()

# === optimizers === #
C_optimizer = optim.Adam(C.parameters(), lr=lr)
O_optimizer = optim.Adam(O.parameters(), lr=lr)
L_optimizer = optim.Adam(L.parameters(), lr=lr)
P_optimizer = optim.Adam(P.parameters(), lr=lr)
J_optimizer = optim.Adam(J.parameters(), lr=lr)

real = 1
fake = 0

# === training === #
C.train()
O.train()
L.train()
P.train()
J.train()

for epoch in range(num_epochs):
	C_running_loss = 0.0
	O_running_loss = 0.0
	L_running_loss = 0.0
	P_running_loss = 0.0
	J_running_loss = 0.0

	for itr, (x) in enumerate(loader):
		# --- label --- #
		real_label = torch.full((batch_size,1), real, device=device)
		fake_label = torch.full((batch_size,1), fake, device=device)

		# --- Update J network --- #
		J_optimizer.zero_grad()

		x_real = x[0].to(device)
		v_real = O(x_real)
		vp_real = P(v_real) + v_real
		vl_real = L(v_real) + v_real
		j_real = J(torch.cat([vp_real, vl_real], dim=1))

		z = torch.randn((batch_size, nz), device=device)
		x_fake = C(z)
		v_fake = O(x_fake)
		vp_fake = P(v_fake) + v_fake
		vl_fake = L(v_fake) + v_fake
		j_fake = J(torch.cat([vp_fake, vl_fake], dim=1))

		loss_real = criterion(j_real, real_label)
		loss_fake = criterion(j_fake, fake_label)
		J_loss = loss_real + loss_fake
		J_loss.backward()
		J_optimizer.step()

		# --- Update C network --- #
		C_optimizer.zero_grad()

		z = torch.randn((batch_size, nz), device=device)
		x_fake = C(z).detach()
		v_fake = O(x_fake)
		vp_fake = P(v_fake) + v_fake
		vl_fake = L(v_fake) + v_fake
		j_fake = J(torch.cat([vp_fake, vl_fake], dim=1))

		loss_real = criterion(j_real, real_label)
		loss_fake = criterion(j_fake, fake_label)

		C_loss = criterion(j_fake, real_label)
		C_loss.backward()
		C_optimizer.step()

		if itr % nj == 0:
			# --- Update P & L network --- #
			P_optimizer.zero_grad()
			L_optimizer.zero_grad()

			x_real = x[0].to(device)
			v_real = O(x_real)
			vp_real = P(v_real) + v_real
			vl_real = L(v_real) + v_real
			j_real = J(torch.cat([vp_real, vl_real], dim=1))

			z = torch.randn((batch_size, nz), device=device)
			x_fake = C(z)
			v_fake = O(x_fake)
			vp_fake = P(v_fake) + v_fake
			vl_fake = L(v_fake) + v_fake
			j_fake = J(torch.cat([vp_fake, vl_fake], dim=1))

			P_loss = torch.mean(j_real) + gamma * torch.mean(torch.norm(vp_real)) + gamma * torch.mean(torch.norm(vp_fake))
			P_loss.backward(retain_graph=True)
			P_optimizer.step()

			L_loss = torch.mean(j_fake) + gamma * torch.mean(torch.norm(vl_real)) + gamma * torch.mean(torch.norm(vl_fake))
			L_loss.backward(retain_graph=True)
			L_optimizer.step()

			# --- Update O network --- #
			O_optimizer.zero_grad()

			x_real = x[0].to(device)
			v_real = O(x_real)
			vp_real = P(v_real)
			vl_real = L(v_real)
			j_real = J(torch.cat([vp_real, vl_real], dim=1))

			z = torch.randn((batch_size, nz), device=device)
			x_fake = C(z)
			v_fake = O(x_fake)
			vp_fake = P(v_fake)
			vl_fake = L(v_fake)
			j_fake = J(torch.cat([vp_fake, vl_fake], dim=1))

			loss_real = criterion(j_real, real_label)
			loss_fake = criterion(j_fake, fake_label)

			O_loss = loss_real + loss_fake
			O_loss.backward()
			O_optimizer.step()

			O_running_loss = O_loss.item()
			L_running_loss = L_loss.item()
			P_running_loss = P_loss.item()
		J_running_loss = J_loss.item()
		C_running_loss = C_loss.item()

	num_itrs = len(dataset) // batch_size
	print('[epoch {:4d}] C: {:.4f}, O: {:.4f}, L: {:.4f}, P: {:.4f}, J: {:.4f}'.format(epoch, C_running_loss / num_itrs, O_running_loss / num_itrs, L_running_loss / num_itrs, P_running_loss / num_itrs, J_running_loss / num_itrs))
	print('real: {:.4f}, fake: {:.4f}'.format(torch.mean(j_real).item(), torch.mean(j_fake).item()))
	writer.add_scalars('Loss', {'C': C_running_loss / num_itrs, 'O': O_running_loss / num_itrs, "L": L_running_loss / num_itrs, "P": P_running_loss / num_itrs, "J": J_running_loss / num_itrs}, global_step=epoch)


# === test === #
C.eval()

generated_samples = []
for i in range(num_test_samples // batch_size):
	z = torch.randn((batch_size, nz), device=device)
	x_fake = C(z)
	generated_samples.append(x_fake)
np_generated_samples = torch.cat(generated_samples).cpu().detach().numpy()

plt.scatter(np_data[:, 0], np_data[:, 1], s=5, c="red")
plt.scatter(np_generated_samples[:, 0], np_generated_samples[:, 1], s=5, c="blue")
plt.savefig(output_filename)