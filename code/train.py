import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

import argparse
import datetime

import numpy as np
import matplotlib.pyplot as plt

import os

from model import Encoder, Decoder
from dataset import RGBDDataset

# Using following code in terminal,
# if you select certain one gpu in multi gpus:
# CUDA_VISIBLE_DEVICES=2 / It means to use GPU#2

def train(args: argparse.ArgumentParser) -> None:
    ## Custom Dataset Settings
    transform = transforms.Compose( [ transforms.ToTensor(), ] )
    
    print("Current Path", args.path)    
    train_dataset = RGBDDataset(args, transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,
                                               shuffle=True,num_workers=2,drop_last=True)
    
    ## Cuda Availabile Check
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    if torch.cuda.is_available():
        print("[SUCCESS] CUDA is available")
        for i in range(torch.cuda.device_count()):
            print(i, torch.cuda.get_device_name(i))
    else:
        print("[FAIL] CUDA is not available")
    
    # Training Settings / Hyper-Parameters
    encoder = Encoder().cuda()
    decoder = Decoder().cuda()
    
    start_epoch = 0
    max_epoch = args.epoch
    
    params = list(encoder.parameters()) + list(decoder.parameters())
    loss_f = nn.MSELoss()
    loss_list = list() # loss graph visualization
    optim_f = torch.optim.Adam(params, lr=0.0002)
    
    # Pre-trained Model Load
    if args.model is not None:
        try:
            encoder, decoder = torch.load(args.model)
            print("[LOAD] Pre-trained model is loaded")
            
            start_epoch+=args.pepoch
            max_epoch+=args.pepoch
        except:
            print("[DEFAULT] Pre-defined model is not loaded")
            print("[Init] Model will be trained from initial parameters")
        
    # Save Directory Settings
    if not os.path.isdir("./model"): os.mkdir("./model")
    if not os.path.isdir("./output"): os.mkdir("./output")
    
    # Model Training
    start_time = datetime.datetime.now()
    
    for i in range(start_epoch, max_epoch):
        epoch_time = datetime.datetime.now()
        for j, [depth, rgb] in enumerate(train_loader):
            optim_f.zero_grad()

            rgb = rgb.clone().detach().cuda()
            #depth = torch.tensor(depth).cuda()
            depth = depth.clone().detach().cuda()
            depth = depth.float()
            output = encoder(depth)
            rgb_like = decoder(output)
            loss = loss_f(rgb_like, rgb)
    
            loss.backward()
            optim_f.step()
    
        #if i % ((max_epoch-start_epoch)/10) == 0:
        if i % 5 == 0 and args.middle_save == True:
            # Output Image Save
            torchvision.utils.save_image(rgb_like.cpu()[0], "./output/epoch_" + str(i) + ".jpg")
            # Model Save
            torch.save([encoder,decoder], "./model/epoch_" + str(i) + ".pth")
        
        loss_list.append(loss.item())
        print("Loss in Epoch", i, loss.item())    
        
        print('Epoch {:4d}/{} Loss : {:.6f}, Runtime : {}'.format(
            i, max_epoch, loss.item(), datetime.datetime.now()-epoch_time))
    
    print("Learning Time :" , datetime.datetime.now() - start_time)
    
    # Trained Model Save
    current_time = str(datetime.datetime.now())[0:16].replace(":", "-").replace(" ", "-")
    path = "./model/"+current_time+"_last-model"
    
    torch.save([encoder, decoder], path+".pth")
    
    # Save Loss Graph
    plt.plot(np.arange(start_epoch, max_epoch), loss_list); plt.savefig(str(current_time)+"_loss-graph.png", dpi=400)