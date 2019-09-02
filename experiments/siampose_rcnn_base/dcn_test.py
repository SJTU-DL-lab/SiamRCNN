from models.DCNv2.dcn_v2 import DCN
import torch
import torch.nn as nn


class DCN_model(nn.Module):
    def __init__(self):
        super(DCN_model, self).__init__()
        self.dcn1 = DCN(64, 64, kernel_size=(3,3), stride=1, padding=1, deformable_groups=1)
        self.dcn2 = DCN(64, 512, kernel_size=(3,3), stride=1, padding=1, deformable_groups=1)
        self.dcn3 = DCN(512, 256, kernel_size=(3,3), stride=1, padding=1, deformable_groups=1)
    
    def forward(self, x):
        output = self.dcn1(x)
        output = self.dcn2(output)
        output = self.dcn3(output)
        
        return output

criterion = torch.nn.MSELoss(reduction='sum')
dcn_model = DCN_model().cuda()
optimizer = torch.optim.SGD(dcn_model.parameters(), lr=0.1)

while True:
    input = torch.randn(2, 64, 256, 256).cuda()
    output = torch.randn(2, 256, 256, 256).cuda()
    # wrap all things (offset and mask) in DCN
    y_pred = dcn_model(input)
    
    loss = criterion(y_pred, output)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(loss)
    


