import os
import argparse
import torch
import torchvision
from torchvision import transforms
import GAN
import matplotlib.pyplot as plt

filepath = os.path.dirname(os.path.abspath(__file__))


def train(G, D, trainloader, optimizer_G, optimizer_D, n_epochs, device, loss_type, sample_size, dataset):
    G.train()  # set to training mode
    D.train()
    generator_losses = []
    discriminator_losses = []
    loss_func = torch.nn.BCELoss()
    for epoch in range(n_epochs):
        g_loss = 0
        d_loss = 0
        for batch_idx, (inputs, x) in enumerate(trainloader):
            G.zero_grad()
            norm_rand = torch.randn(trainloader.batch_size, G.latent_dim).to(device)
            generator_out = G(norm_rand)
            discriminator_out = D(generator_out)
            generator_loss = loss_func(discriminator_out, torch.ones(trainloader.batch_size, 1).to(device))
            if loss_type == 'original':
                generator_loss *= -1
            generator_loss.backward()
            optimizer_G.step()
            g_loss += generator_loss.item()

            # train the discriminator twice for each time you train the generator
            for i in range(2):
                D.zero_grad()
                # discriminator trains on generated images
                norm_rand = torch.randn(trainloader.batch_size, G.latent_dim).to(device)
                generator_out = G(norm_rand)
                discriminator_out_generated = D(generator_out)
                d_loss_generated = loss_func(discriminator_out_generated, torch.zeros(trainloader.batch_size, 1).to(device))
                # discriminator trains on real images
                data_real = inputs.view(-1, 784).to(device)
                discriminator_out_real = D(data_real)
                d_loss_real = loss_func(discriminator_out_real,
                                        torch.ones(data_real.size(0), 1).to(device))
                discriminator_loss = d_loss_real + d_loss_generated
                discriminator_loss.backward()
                optimizer_D.step()
                d_loss += discriminator_loss.item()

        generator_losses.append(g_loss)
        discriminator_losses.append(d_loss)

        print("Epoch: {}, Generator Loss: {}, Discriminator Loss: {}".format(epoch, g_loss, d_loss))
        if (epoch + 1) % 10 == 0:
            sample(G, index=epoch+1, sample_size=sample_size, device=device, dataset=dataset, loss_type=loss_type)

    return generator_losses, discriminator_losses


def sample(G, index, sample_size, device, dataset, loss_type):
    G.eval()  # set to inference mode
    with torch.no_grad():
        norm_rand = torch.randn(sample_size, G.latent_dim).to(device)
        generated = G(norm_rand)
        torchvision.utils.save_image(generated.view(generated.size(0), 1, 28, 28), filepath +
                                     '/samples/sample_' + dataset + '_epoch_' + str(index)
                                     + 'loss_' + str(loss_type) + '.png')


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
                                              train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=0)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
                                             train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=0)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
                                                     train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=0)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                                    train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=0)
    else:
        raise ValueError('Dataset not implemented')

    filename = '%s_' % args.dataset \
               + 'batch%d_' % args.batch_size \
               + 'mid%d_' % args.latent_dim

    G = GAN.Generator(latent_dim=args.latent_dim,
                      batch_size=args.batch_size, device=device).to(device)
    D = GAN.Discriminator().to(device)
    optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr)
    g_loss, d_loss = train(G, D, trainloader, optimizer_G, optimizer_D,
                                args.epochs, device, args.loss, args.sample_size, args.dataset)
    plt.figure()
    plt.plot(g_loss)
    plt.plot(d_loss)
    plt.legend(['Generator Loss', 'Discriminator Loss'])
    plt.title('Loss vs. Epoch')
    plt.savefig(filepath + "/loss/" + f"{args.dataset}_{args.loss}_loss.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=100)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--latent-dim',
                        help='.',
                        type=int,
                        default=100)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=2e-4)
    parser.add_argument('--loss',
                        help='maximize standard loss or minimize modified loss',
                        type=str,
                        default='original')

    args = parser.parse_args()
    main(args)
