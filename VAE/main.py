import os
import torch
from torch import nn
from PIL import Image
from tqdm import tqdm
from vae import Encoder, Decoder
from matplotlib import pyplot as plt
from torchvision import datasets, transforms

epoch = 10
batch_size = 32
hidden_dim = 200

if __name__ == '__main__':

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = datasets.MNIST(root='./VAE/data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    encoder = Encoder(28 * 28, 500, hidden_dim * 2)
    decoder = Decoder(hidden_dim, 500, 28 * 28 * 2)

    encoder.to('cuda')
    decoder.to('cuda')

    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=2e-4)
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=2e-4)
    recon_loss = nn.MSELoss()

    losses = []

    for i in range(epoch):
        bar = tqdm(train_loader, desc=f'epoch {i}')
        for X, y in bar: # X: [bs, 1, 28, 28]
            X = X.to('cuda')
            bs = X.shape[0]
            # calculate z (encode)
            
            noise = torch.randn(bs, hidden_dim).to('cuda') # [bs, hidden_dim]
            pred_enc_param = encoder(X.reshape(X.shape[0], -1)) # [bs, hidden_dim * 2]
            mu, logvar = pred_enc_param[:, :hidden_dim], pred_enc_param[:, hidden_dim:] # [bs, hidden_dim], [bs, hidden_dim]
            sigma = torch.exp(0.5 * logvar)
            z = mu + sigma * noise # [bs, hidden_dim]

            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            loss1 = kl_divergence.mean()

            # calculate x'(decode)
            noise = torch.randn(bs, 28 * 28).to('cuda') # [bs, 28 * 28]
            pred_dec_param = decoder(z) # [bs, 28 * 28 * 2]
            x_mu, x_logvar = pred_dec_param[:, :28*28], pred_dec_param[:, 28*28:]  # Change x_sigma to x_logvar
            x_sigma = torch.exp(0.5 * x_logvar)  # Calculate x_sigma from x_logvar
            pred_x = x_mu + x_sigma * noise # [bs, 28 * 28]

            loss2 = recon_loss(pred_x, X.reshape(X.shape[0], -1))
            loss = loss1 + loss2

            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()

            loss.backward()
            losses.append(loss.item())

            enc_optimizer.step()
            dec_optimizer.step()

            bar.set_description(f'epoch {i}, loss={loss}, loss1={loss1}, loss2={loss2}')
    
    z = torch.randn(1, hidden_dim).to('cuda')
    pred_dec_param = decoder(z) # [bs, 28 * 28 * 2]
    x_mu, x_logvar = pred_dec_param[:, :28*28], pred_dec_param[:, 28*28:]  # Change x_sigma to x_logvar
    print(x_mu.shape)
    x_sigma = torch.exp(0.5 * x_logvar)  # Calculate x_sigma from x_logvar
    noise = torch.randn(1, 28 * 28).to('cuda')
    pred_x = x_mu + x_sigma * noise # [bs, 28 * 28]
    print(pred_x.shape)
    pred_x = pred_x.reshape(1, 28, 28)
    inverse_transform = transforms.Compose([transforms.Normalize(mean=(-1), std=(2)), transforms.ToPILImage()])
    pred_x = inverse_transform(pred_x)
    
    plt.imshow(pred_x, cmap='gray')
    plt.axis('off')
    plt.show()