import math
import numpy as np
import os
import matplotlib.pyplot as plt
from vis_gpu import mb_monotone
from numba import cuda
from PIL import Image

np_path = os.path.join(os.path.dirname(__file__), f'data/toystar_pos_2d.npy')

with open(np_path,'rb') as f:
    all_pos = np.load(f)

img_dim = 1024
threads = 16
tpb = (threads, threads)
bpg = ( int(img_dim / threads), int(img_dim / threads) )
image = cuda.to_device(np.zeros((img_dim, img_dim), dtype='f4'))
mb_threshold = 11.0#9.0 # 9.0 for 3d->2d projection, 400 particles
mb_radius = 0.02
view_size = 2.8

downsample = 3

all_pos = all_pos[:,:,::downsample]

all_images = np.zeros((img_dim, img_dim, all_pos.shape[2]), dtype='f4')

for frame_idx in range(all_pos.shape[2]):

    print(frame_idx)

    frame = all_pos[:,:,frame_idx]
    points = cuda.to_device(frame.copy())
    mb_monotone[bpg, tpb](points, mb_threshold, mb_radius, view_size, image)
    all_images[:,:,frame_idx] = image.copy_to_host()


images_color = np.zeros((img_dim, img_dim, all_images.shape[2], 3), dtype=np.uint8)

img_g = np.zeros((img_dim, img_dim, all_images.shape[2]), dtype=np.uint8)
img_b = np.zeros((img_dim, img_dim, all_images.shape[2]), dtype=np.uint8)

img_g[all_images == 1.0] =  150
# img_g[all_images == 1.0] =  255
img_b[all_images == 1.0] =  255

images_color[:,:,:,1] = img_g
images_color[:,:,:,2] = img_b
    
imgs = [Image.fromarray(images_color[:,:,i]) for i in range(images_color.shape[2])]
mb_path = os.path.join(os.path.dirname(__file__), f'figures/02_vis_toystar/mb_toystar_vis.gif')
imgs[0].save(mb_path, save_all=True, append_images=imgs[1:], duration=35, loop=0)
