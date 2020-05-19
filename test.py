"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os, torch
from collections import OrderedDict

from torch.nn import functional as F
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from torchvision import utils
from data.base_dataset import get_params, get_transform
from PIL import Image
from pathlib import Path

from models.networks.model import BiSeNet
import torchvision.transforms as transforms

def vis_condition_img(img):
    part_colors = torch.tensor([[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]).float()
    N,C,H,W = img.size()
    condition_img_color = torch.zeros((N,3,H,W))
    num_of_class = int(torch.max(img))
    for pi in range(1, num_of_class + 1):
        index = (img == pi).nonzero()
        condition_img_color[index[:,0],:,index[:,2], index[:,3]] = part_colors[pi]
    condition_img_color = condition_img_color/255*2.0-1.0
    return condition_img_color

def save_fig(result,save_root,count,nrows):
    utils.save_image(
        result,
        f'{save_root}/{str(count).zfill(6)}.png',
        nrow=nrows,
        normalize=True,
        range=(-1, 1),
    )

def load_label(label_path):
    label = Image.open(label_path)
    params = get_params(opt, label.size)
    transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
    label_tensor = transform_label(label) * 255.0
    label_tensor[label_tensor == 255] = opt.label_nc  # 'unknown' is opt.label_nc
    return label_tensor.unsqueeze(0)

def load_img(img_path):
    image = Image.open(img_path)
    image = image.convert('RGB')

    params = get_params(opt, image.size)
    transform_image = get_transform(opt, params)
    image_tensor = transform_image(image)
    return image_tensor.unsqueeze(0)

def parsing_img(bisNet, img):
    with torch.no_grad():
        im_img = F.interpolate(img, size=(512, 512), mode='bilinear',align_corners=False)
        seg_label = torch.argmax(bisNet(im_img)[0],1,keepdim=True)

    return seg_label

def initFaceParsing():
    net = BiSeNet(n_classes=19)
    net.cuda()
    net.load_state_dict(torch.load('E:/face2face2/code/face-parsing/res/cp/79999_iter.pth'))
    net.eval()
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return net,to_tensor

opt = TestOptions().parse()

# dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

save_root =  os.path.join(opt.results_dir, opt.name)
if not os.path.exists(save_root):
    os.mkdir(save_root)


label_path = sorted(os.listdir(opt.label_dir))
label_num = min(len(label_path), opt.how_many)
print("Synthesize {} images".format(label_num))
# random seg
if 0 == opt.MODE:
    '''
    # --label_dir E:/face2face2/test/segs --results_dir F:/dgene/SPADE/results/ --use_vae --no_instance --MODE 0
    '''
    result,seg_labels,nrows,count = [],[],1, 0
    save_root = os.path.join(save_root,'randomSeg',"epoch_{}".format(opt.which_epoch))
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    
    
    with torch.no_grad():
        for i, file in enumerate(label_path[:label_num]):
            if (not file.endswith('g')):
                continue

            img_path = os.path.join(opt.label_dir,file)
            label_tensor = load_label(img_path)
            seg_labels.append(label_tensor)
            #result.append(vis_condition_img(label_tensor))
            if nrows-1 == i%nrows or i==len(label_path)-1:
                seg_labels = torch.cat(seg_labels, dim=0).cuda()
                data_i = {'label': seg_labels, 'instance': torch.tensor([0]), 'image': torch.tensor([0])}
                input_semantics, _ = model.preprocess_input(data_i)

                batch = seg_labels.size(0)
                #result = [torch.cat(result, dim=0)]
                for j in range(1):
                    #noise = torch.randn(1, opt.z_dim, dtype=torch.float32,device=data_i['label'].get_device()).repeat(batch,1)
                    #fake_image = model.netG(input_semantics, z=noise)
                    fake_image = model.netG(input_semantics)
                    result.append(fake_image.detach().cpu())

                result = torch.cat(result, dim=0)
                save_fig(result, save_root, count, batch)
                count += 1
                result,seg_labels = [],[]

from pathlib import Path
# guide style
if 1 == opt.MODE:
    '''
    # --image_dir E:/face2face2/dataSet/CelebAMask-HQ/image --label_dir E:/face2face2/test/segs --results_dir F:/dgene/SPADE/results/ --use_vae --no_instance --MODE 1
    '''
    result,seg_labels,nrows,style_num,count = [],[],1, 1, 0
    save_root = os.path.join(save_root,'guideStyle', "epoch_{}".format(opt.which_epoch))
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    label_path = sorted(os.listdir(opt.label_dir))
    # style_img_path = sorted(os.listdir(opt.image_dir))
    style_img_path = sorted(Path(opt.image_dir).rglob('*.png'))
    
    print("style img len is {}".format(len(style_img_path)))
    with torch.no_grad():
        #result.append(-torch.ones(1,3,opt.crop_size,opt.crop_size))
        for i, file in enumerate(label_path[:label_num]):
            if (not file.endswith('g')):
                continue

            img_path = os.path.join(opt.label_dir,file)
            label_tensor = load_label(img_path)
            seg_labels.append(label_tensor)
            #result.append(vis_condition_img(label_tensor))
            if nrows-1 == i%nrows or i==len(label_path)-1:
                seg_labels = torch.cat(seg_labels, dim=0).cuda()
                data_i = {'label': seg_labels, 'instance': torch.tensor([0]), 'image': torch.tensor([0])}
                input_semantics, _ = model.preprocess_input(data_i)

                index = torch.randperm(len(style_img_path)).long()[:style_num]
                # load style images
                style_img = []
                for item in index:
                    img_path = os.path.join(opt.image_dir, style_img_path[item])
                    style_img.append(load_img(img_path))
                z, _, _ = model.encode_z(torch.cat(style_img, dim=0).cuda())

                batch = seg_labels.size(0)
                #result = [torch.cat(result, dim=0)]
                for j in range(style_num):
                    noise = z[[j]].repeat(batch,1)
                    fake_image = model.netG(input_semantics, z=noise)

                    #result.append(style_img[j])
                    result.append(fake_image.detach().cpu())

                result = torch.cat(result, dim=0)
                save_fig(result, save_root, count, batch+1)
                count += 1
                result,seg_labels = [],[]
                #result.append(-torch.ones(1, 3, opt.crop_size, opt.crop_size))

# guide style and seg on style image
if 2 == opt.MODE:
    '''
    # --image_dir E:/face2face2/dataSet/CelebAMask-HQ/image --results_dir F:/dgene/SPADE/results/ --use_vae --no_instance --MODE 2
    # --image_dir L:/chenanpei/databased/images/0000 --results_dir F:/dgene/SPADE/results/ --use_vae --no_instance --MODE 2
    '''
    result,style_img,nrows,style_num,count = [],[],7, 5, 0
    save_root = os.path.join(save_root,'guideStyle2')
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    bisNet, _ = initFaceParsing()
    style_img_path = sorted(os.listdir(opt.image_dir))
    index = torch.randperm(len(style_img_path)).long()[:25]
    with torch.no_grad():
        result.append(-torch.ones(1,3,opt.crop_size,opt.crop_size))
        for i, item in enumerate(index):

            img_path = os.path.join(opt.image_dir, style_img_path[item])
            style_img.append(load_img(img_path))

            if nrows-1 == i%nrows or i==len(index)-1:
                style_img = torch.cat(style_img,dim=0).cuda()
                z, _, _ = model.encode_z(style_img)

                seg_labels = parsing_img(bisNet,style_img)
                data_i = {'label': seg_labels, 'instance': torch.tensor([0]), 'image': torch.tensor([0])}
                input_semantics, _ = model.preprocess_input(data_i)

                batch = seg_labels.size(0)
                result.append(vis_condition_img(seg_labels.cpu()))
                for j in range(batch):
                    noise = z[[j]].repeat(batch,1)
                    fake_image = model.netG(input_semantics, z=noise)

                    result.append(style_img[[j]].cpu())
                    result.append(fake_image.detach().cpu())

                result = torch.cat(result, dim=0)
                save_fig(result, save_root, count, batch+1)
                count += 1
                result,style_img = [],[]
                result.append(-torch.ones(1, 3, opt.crop_size, opt.crop_size))

# # random noise
# elif 1 == opt.MODE:
#     nrows = 4
#     save_root = os.path.join(save_root,'randomStyle')
#     if not os.path.exists(save_root):
#         os.mkdir(save_root)
#
#     with torch.no_grad():
#         result, count = [], 0
#         for i, data_i in enumerate(dataloader):
#             if i * opt.batchSize >= opt.how_many:
#                 break
#
#             img_path = data_i['path']
#             print('process image... %s' % img_path[0])
#             input_semantics, _  = model.preprocess_input(data_i)
#             result.append(vis_condition_img(data_i['label']).detach().cpu())
#             for j in range(nrows):
#                 noise = torch.randn(opt.batchSize, opt.z_dim,dtype=torch.float32, device=data_i['label'].get_device())
#                 fake_image = model.netG(input_semantics, z=noise)
#                 result.append(fake_image.detach().cpu())
#
#
#             if nrows*(nrows+1) == len(result) or i==len(dataloader):
#                 result = torch.cat(result, dim=0)
#                 save_fig(result, save_root, count, nrows+1)
#                 count += 1
#                 result = []
#
# elif 2 == opt.MODE:
#     nrows = 5
#     save_root = os.path.join(save_root,'mul-view')
#     if not os.path.exists(save_root):
#         os.mkdir(save_root)
#
#     noise = torch.randn(opt.batchSize, opt.z_dim, dtype=torch.float32).cuda()
#     with torch.no_grad():
#         result, count = [], 0
#         for i, data_i in enumerate(dataloader):
#
#             img_path = data_i['path']
#             print('process image... %s' % img_path[0])
#             input_semantics, _  = model.preprocess_input(data_i)
#
#             fake_image = model.netG(input_semantics, z=noise)
#             result.append(fake_image.detach().cpu())
#
#             if nrows**2 == len(result) or i==len(dataloader):
#                 result = torch.cat(result, dim=0).flip(3)
#                 save_fig(result, save_root, count, nrows)
#                 count += 1
#                 result = []
