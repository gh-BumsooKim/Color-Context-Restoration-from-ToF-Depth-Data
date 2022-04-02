import torch
from torchvision import transforms

import numpy as np
import cv2

import argparse
import datetime

import open3d as o3d

def test(args: argparse.ArgumentParser) -> None:
    
    if args.model == None:
        raise RuntimeError("[Error] Pre-trained model is not loaded")
    else:
        encoder, decoder = torch.load(args.model)
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    
    transform = transforms.Compose( [ transforms.ToTensor(), ] )
    
    # Kinect Connection
    kinect = o3d.io.AzureKinectSensor(o3d.io.AzureKinectSensorConfig())
    
    if not kinect.connect(0):
        raise RuntimeError('[FAIL] Failed to connect to sensor')
        
    while True:
        
        # Image Input
        start_time = datetime.datetime.now()
        rgbd = kinect.capture_frame(True)
        if rgbd is None:
            continue
        
        input_raw = np.asarray(rgbd.depth, dtype=np.float32)
        input_image = transform(input_raw).unsqueeze(0).cuda()
        #print(input.shape) -> (1, 720, 1280)
        
        # Image Reconstruction in AutoEncoder
        with torch.no_grad():
            output = encoder(input_image)
            output = decoder(output)
        
        #print(output.shape) -> (3, 720, 1280)
        output_image = output.cpu().squeeze()
        output_image = output_image.numpy().transpose(1,2,0)
        output_image[:,:,:] = output_image[:,:,::-1]
        
        # Image Show with Raw Depth Map
        print("Output Image Shape :", output_image.shape)
        
        input_raw = np.expand_dims(input_raw, axis=2)
        input_raw = np.concatenate((input_raw, input_raw, input_raw), axis=2)
        
        window_image = np.hstack([input_raw, output_image])
        cv2.imshow("Depth to Color", window_image)
        print(datetime.datetime.now() - start_time)
        
        key = cv2.waitKey(33)
        if  key & 0xFF == ord('q'):
            cv2.destroyAllWindows()
        