import torch
import torch.nn as nn

class vae(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28,256),
            nn.ReLU(),
        )

        self.mu = nn.Linear(256,128)
        self.logvar = nn.Linear(256,128)

        self.decoder = nn.Sequential(
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,28*28),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, ):
        x = x.view(-1,28*28)

        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        out = self.decoder(z)

        return out, mu, logvar

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)





class gvae(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10,16)

        self.encoder = nn.Sequential(
            nn.Linear(28*28+16,256),
            nn.ReLU(),
        )

        self.mu = nn.Linear(256,128)
        self.logvar = nn.Linear(256,128)

        self.decoder = nn.Sequential(
            nn.Linear(128+16,256),
            nn.ReLU(),
            nn.Linear(256,28*28),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)
    def decode(self, z,labels):
        labels =self.embedding(labels)
        z=torch.cat([z,labels],dim=1)
        return self.decoder(z)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, labels):
        x = x.view(-1,28*28)
        labels = self.embedding(labels)

        enc_in = torch.cat([x, labels], dim=1)
        mu, logvar = self.encode(enc_in)

        z = self.reparameterize(mu, logvar)
        dec_in = torch.cat([z, labels], dim=1)
        out = self.decoder(dec_in)

        return out, mu, logvar

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
