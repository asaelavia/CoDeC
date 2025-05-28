import torch
from torch import nn, optim


class Mlp(nn.Module):
    def __init__(self, input_dim, hidden_dims=None):
        super(Mlp, self).__init__()
        if hidden_dims is None:
            hidden_dims = [64, 10]
        model_lis = []
        hidden_dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims) - 1):
            model_lis += [nn.Linear(hidden_dims[i], hidden_dims[i + 1])]
            model_lis += [nn.ReLU()]
        model_lis.pop()
        model_lis += [nn.Sigmoid()]
        self.model = nn.Sequential(*model_lis)

    def forward(self, input):
        return self.model(input)

def pretrain(mlp, device, train_dl, lr=1e-3, epochs=100, save=True):
    mlp.train()
    optimizer = optim.Adam(mlp.parameters(), lr=lr)
    loss = nn.BCELoss()
    for e in range(epochs):
        tot_size = 0
        tot_loss_1 = 0
        train_acc = 0
        for data in train_dl:
            # img, clas, y_gt = data
            img, clas = data
            # img, clas,y_gt,indexes = data
            tot_size += len(clas)
            img = img.to(device)
            clas = clas.to(device)
            clas = clas.squeeze()
            # clas = clas.long()
            optimizer.zero_grad()

            # bsz = clas.shape[0]

            # y = encoder_source(img)
            y = mlp(img)
            ls_1 = loss(y.reshape(y.shape[0],), clas.float())
            # ls_1 = loss(y, clas,indexes)
            ls_1.backward()
            tot_loss_1 += ls_1.item()
            optimizer.step()
            y = torch.round(y)

            # rights = torch.sum(y.detach().cpu() == y_gt)
            rights = torch.sum(y.detach().reshape(y.shape[0],).cpu() == clas.cpu())
            train_acc += rights.item()
            del img, clas
        print(f'Encoder Train Acc {train_acc / tot_size}')
        los_1 = tot_loss_1 / tot_size
        print(f'epoch {e} loss with encoder is {los_1}')
        # if e % 100 == 0:
        #     if save:
        # os.makedirs(f'./models/{source}_{target}', exist_ok=True)
        # torch.save(mlp.state_dict(), f'./models/{source}_{target}/pre_mlp_{e}')
        mlp.eval()
        mlp.train()
    # if save:
    # torch.save(mlp.state_dict(), f'./models/{source}_{target}/pre_mlp_{epochs}')
    mlp.eval()
    mlp.train()
    return mlp