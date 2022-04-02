import torch.nn as nn

## Model Definition
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.layer1 = nn.Sequential(
            # batch x 1 x 720 x 1280
            nn.Conv2d(1,16,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2,2),              
            # batch x 16 x 360 x 640
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2),              
            # batch x 32 x 180 x 320
            nn.Conv2d(32,64,3,padding=1),  
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2)               
            # batch x 64 x 90 x 160
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),  
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            # batch x 128 x 45 x 80
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # batch x 256 x 45 x 80
            nn.Conv2d(256,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # batch x 512 x 45 x 80
            nn.Conv2d(512,1024,3,padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
            # batch x 1024 x 45 x 80
        )
        
    def forward(self, x):
        # x = batch x 1 x 720 x 1280       
        out = self.layer1(x)
        # out=batch x 64 x 90 x 160
        out = self.layer2(out)
        # out=batch x 1024 x 45 x 80
        return out
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
            
        self.layer1 = nn.Sequential(
            # batch x 1024 x 45 x 80
            nn.ConvTranspose2d(1024,512,3,1,1),     
            nn.ReLU(),
            nn.BatchNorm2d(512),
            # batch x 512 x 45 x 80
            nn.ConvTranspose2d(512,256,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # batch x 256 x 45 x 80
            nn.ConvTranspose2d(256,128,3,2,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # batch x 128 x 90 x 160
            nn.ConvTranspose2d(128,64,3,2,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
            # batch x 64 x 180 x 320
        )
        
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64,16,3,2,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            # batch x 16 x 360 x 640
            nn.ConvTranspose2d(16,3,3,2,1,1),
            nn.ReLU()
            # batch x 3 x 720 x 1280
        )
        
    def forward(self, x):
        # x = batch x 1024 x 45 x 80
        out = self.layer1(x)
        # out=batch x 64 x 180 x 320
        out = self.layer2(out)
        # out=batch x 3 x 720 x 1280
        return out