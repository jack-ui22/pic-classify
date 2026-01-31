import torch
import numpy as np
import matplotlib.pyplot as plt
from model import vae

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = vae().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

num_generate = 10
latent_dim = 128
label = 8

with torch.no_grad():

    z = torch.randn(num_generate, latent_dim).to(device)


    labels = torch.full((num_generate,), label, dtype=torch.long).to(device)
    labels_emb = model.embedding(labels)
    print(labels_emb)

    dec_in = torch.cat([z, labels_emb], dim=1)
    gen_imgs_flat = model.decoder(dec_in)

    gen_imgs = gen_imgs_flat.view(num_generate, 28, 28).cpu().numpy()

plt.figure(figsize=(15,3))
for i in range(num_generate):
    plt.subplot(1, num_generate, i+1)
    plt.imshow(gen_imgs[i], cmap="gray")
    plt.axis("off")
plt.suptitle(f"CVAE_num: {label}")
plt.show()
