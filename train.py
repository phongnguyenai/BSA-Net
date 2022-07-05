import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from torch_geometric.data import DataLoader, Batch, Data

from BSA_model import BSANet
from chamfer_distance import ChamferDistance

torch.cuda.empty_cache()

class Dataset(Dataset):
    def __init__(self, pc_in_file, pc_out_file, img_file, transform=None):
        self.imgs = np.load(img_file)
        self.pcs_in = np.load(pc_in_file)
        self.pcs_out = np.load(pc_out_file)
        self.transform = transform

    def __len__(self):
        return self.pcs_in.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img = self.imgs[idx]
        img = torch.Tensor(img)
        img = img.permute(2,0,1)
        
        pc_in = self.pcs_in[idx]
        pc_in = torch.Tensor(pc_in)
        pc_in = Data(pos=pc_in)
        
        pc_out = self.pcs_out[idx]
        pc_out = torch.Tensor(pc_out)
        pc_out = Data(pos=pc_out)
        
        sample = {"pc_in": pc_in, "pc_out": pc_out, 'img': img}

        if self.transform:
            sample = self.transform(sample)

        return sample

def evaluation():
    model.eval()
    total_loss = 0
    
    for data in test_dataloader:      
        pc_out = data['pc_out']
        pc_out = pc_out.to(device)
        
        with torch.no_grad():
            decoded = model(data)
            dist1, dist2 = criterion(decoded.reshape(-1,2048,3), pc_out.pos.reshape(-1,2048,3))
            loss = (torch.mean(dist1)) + (torch.mean(dist2))
            total_loss += loss.item() * pc_out.num_graphs
            
    return total_loss/len(test_set)

def train():
    model.train()
    total_loss = 0
    
    for data in train_dataloader:
        pc_out = data['pc_out']
        pc_out = pc_out.to(device)
        
        optimizer.zero_grad()
        decoded = model(data)
        
        dist1, dist2 = criterion(decoded.reshape(-1,2048,3), pc_out.pos.reshape(-1,2048,3))
        loss = (torch.mean(dist1)) + (torch.mean(dist2))
        
        loss.backward()
        total_loss += loss.item() * pc_out.num_graphs
        optimizer.step()
        
    return total_loss / len(train_set)

# Data preparation
with open('data.pickle', 'rb') as handle:
    b = pickle.load(handle)
    
train_set = b['train']
test_set = b['test']

batch_size = 4

train_dataloader = DataLoader(train_set, batch_size=batch_size,
                    shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=batch_size,
                    shuffle=True)

# Choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# New training
model = BSANet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = ChamferDistance()

# Incremental training
# model = BSANet().to(device)
# model.load_state_dict(torch.load("pretrained/BSA-Net_1000_5.7170_5.6984.pt"))
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = ChamferDistance()
        
# Start training
for epoch in range(1001):
    train_loss = train()
    print('Epoch {:03d}, Training loss: {:.4f}'.format(epoch, train_loss))

    if epoch % 100 == 0:
        eval_loss = evaluation()
        print('Epoch {:03d}, Evaluation loss: {:.4f}'.format(epoch, eval_loss))
        torch.save(model.state_dict(),'./pretrained/BSA-Net_{}_{:.4f}_{:.4f}.pt'.format(epoch,train_loss,eval_loss))