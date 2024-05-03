import torch
from model import ColorEquivariantSlotAttentionNet

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

model = ColorEquivariantSlotAttentionNet(resolution=(128, 128), 
                                             num_slots=7, 
                                             num_iterations=3, 
                                             hid_dim=32,
                                             planes=20,
                                             rotations=3, 
                                             separable=True,
                                             ce_layers=4).to(device)

model.load_state_dict(torch.load('./tmp/run2/model0.ckpt')['model_state_dict'])