import torch 
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def load_image(img : Image, device, resize_dims=(512,512)):
    # take any image and return a latent to put into vae encoder 
    img = img.convert("RGB")
    img = img.resize(resize_dims)
    img = 2.0 * np.array(img).astype(np.float32) / 255.0 - 1.0
    img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    return img  # size [1,3,512,512]

def load_image_batch(imags, device, resize_dims=(512,512)):
    imgs_batch = []
    for img in imags:
        img = load_image(img, device, resize_dims)
        imgs_batch.append(img)
    imgs_batch = torch.cat(imgs_batch, dim=0)
    return imgs_batch

def output_image(img):
    # take latent of size [1,3,512,512] and return PIL image
    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img + 1.0) / 2.0
    img = np.clip(img, 0.0, 1.0)
    img = (255 * img).astype(np.uint8)
    return Image.fromarray(img)

def output_image_batch(imgs):
    # take latent of size [N,3,512,512] and return PIL image list
    imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
    imgs = (imgs + 1.0) / 2.0
    imgs = np.clip(imgs, 0.0, 1.0)
    imgs = (255 * imgs).astype(np.uint8)
    img_list = [Image.fromarray(img) for img in imgs]
    return img_list

def display_alongside(img_list, resize_dims=(512,512), padding=10, frame_color=(255, 255, 255)):
    # take a list of PIL images and return a single image with all of them displayed alongside each other
    padded_width = resize_dims[0] + 2 * padding
    padded_height = resize_dims[1] + 2 * padding
    res = Image.new("RGB", (padded_width * len(img_list), padded_height), frame_color)
    for i, img in enumerate(img_list):
        x_offset = i * padded_width + padding
        y_offset = padding
        img_resized = img.resize(resize_dims)
        res.paste(img_resized, (x_offset, y_offset))
    return res

def display_in_two_rows(img_list, resize_dims=(512,512), padding=5, frame_color=(255, 255, 255)):
    # Number of images in each row
    num_images = len(img_list)
    num_images_per_row = (num_images + 1) // 2  # Divide images into two rows, rounding up for the first row
    padded_width = resize_dims[0] + 2 * padding
    padded_height = resize_dims[1] + 2 * padding
    total_width = padded_width * num_images_per_row
    total_height = padded_height * 2  # Two rows
    res = Image.new("RGB", (total_width, total_height), frame_color)    
    for i, img in enumerate(img_list):
        row = i // num_images_per_row
        col = i % num_images_per_row
        x_offset = col * padded_width + padding
        y_offset = row * padded_height + padding
        img_resized = img.resize(resize_dims)
        res.paste(img_resized, (x_offset, y_offset))
    return res


def lerp_cond_embed(ts, embed_cond_A, embed_cond_B):
    # linear interpolation between two conditional embeddings
    return torch.cat([(1 - t) * embed_cond_A + t * embed_cond_B for t in ts], dim=0) 

def o_project(x, y):
    # project torch vector x onto the orthogonal complement of y
    y_hat = y / torch.norm(y)
    return x - torch.dot(x, y_hat) * y_hat

def o_project_(xs, ys):
    # project numpy batch vector x onto the orthogonal complement of y
    ys_norm = torch.norm(ys, dim=-1)
    ys_hat = ys / ys_norm[:, None]
    return xs - torch.sum(xs * ys_hat, dim=-1)[:,None] * ys_hat

def norm_fix(x, m):
    # maintain the tensor x norm as m
    return m * x / torch.norm(x)

def norm_fix_(xs, m):
    # maintain the batch tensor xs norm as m
    if isinstance(m, float):
        m = torch.tensor([m] * xs.shape[0]).to(xs.device)
    return m[:,None] * xs / torch.norm(xs, dim=-1)[:, None]



def pca_latent(input):
    if isinstance(input, torch.Tensor):
        input = input.cpu().detach().numpy()
    if input.shape[0] == 1:
        input = input[0] # 4*64*64
    input = np.transpose(input, (1, 2, 0))
    input = input.reshape(-1, 4)  # (4096, 4)
    pca = PCA(n_components=1)  # Reduce to 1 principal components for visualization
    principal_components = pca.fit_transform(input)
    principal_component = principal_components[:,0]
    output = principal_component.reshape(64, 64)
    return output


def plot_heatmap(matrix, show=True, save_path=None):
    plt.figure(figsize=(7, 5))
    sns.heatmap(matrix, cmap='viridis')
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path)
        print('Save heatmap to ', save_path)
    plt.close()


