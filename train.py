import os
import argparse
from torchvision import transforms
from model import *
from tqdm import tqdm
import time
import datetime
import torch.optim as optim
import torch

from PIL import Image
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument
parser.add_argument('--model_dir', default='./tmp/model10.ckpt', type=str, help='where to save models' )
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_slots', default=7, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
parser.add_argument('--hid_dim', default=32, type=int, help='hidden dimension size')
parser.add_argument('--learning_rate', default=0.0004, type=float)
parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--num_epochs', default=31, type=int, help='number of workers for loading data')
parser.add_argument('--planes', default=20, type=int)
parser.add_argument('--rotations', default=3, type=int)
parser.add_argument('--seperable', default=False, type=bool)


opt = parser.parse_args()
resolution = (128, 128)

directory = '/DATA/aditya/test_creation_reshaped/image_task1'

class CustomImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_files = [f for f in os.listdir(directory) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_set = CustomImageDataset(directory, transform=transform)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# train_set = CLEVR('train')

# model = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.num_iterations, opt.hid_dim).to(device)

model = ColorEquivariantSlotAttentionNet(resolution=(128, 128), 
                                             num_slots=opt.num_slots, 
                                             num_iterations=opt.num_iterations, 
                                             hid_dim=opt.hid_dim,
                                             planes=opt.planes,
                                             rotations=opt.rotations, 
                                             separable=opt.seperable,
                                             ce_layers=4).to(device)
# model.load_state_dict(torch.load('./tmp/model6.ckpt')['model_state_dict'])

criterion = nn.MSELoss()

params = [{'params': model.parameters()}]

train_dataloader = DataLoader(train_set, batch_size=opt.batch_size,
                        shuffle=True, num_workers=opt.num_workers)

optimizer = optim.Adam(params, lr=opt.learning_rate)

start = time.time()
i = 0

for epoch in range(opt.num_epochs):
    model.train()

    total_loss = 0

    for sample in tqdm(train_dataloader):
        i += 1

        if i < opt.warmup_steps:
            learning_rate = opt.learning_rate * (i / opt.warmup_steps)
        else:
            learning_rate = opt.learning_rate

        learning_rate = learning_rate * (opt.decay_rate ** (
            i / opt.decay_steps))

        optimizer.param_groups[0]['lr'] = learning_rate
        
        image = sample.to(device)
        recon_combined, recons, masks, slots = model(image)
        loss = criterion(recon_combined, image)
        total_loss += loss.item()

        del recons, masks, slots

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_loss /= len(train_dataloader)

    print ("Epoch: {}, Loss: {}, Time: {}".format(epoch, total_loss,
        datetime.timedelta(seconds=time.time() - start)))

    if not epoch % 5:
        torch.save({
            'model_state_dict': model.state_dict(),
            }, f'./tmp/model{epoch}.ckpt')
