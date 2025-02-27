import random
import numpy as np
import torch
import torch_geometric as pyg
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm.auto import *

import deepgd as dgd
from fa2.run_fa2 import run_fa2

if __name__ == '__main__':

    # enter a string of the prev model name to continue training
    prev_model = None

    device = "cuda"
    for backend, device_name in {
        torch.backends.mps: "mps",
        torch.cuda: "cuda",
    }.items():
        if backend.is_available():
            device = device_name

    batch_size = 24
    lr = 0.0005
    dataset = dgd.RomeDataset(layout_initializer = run_fa2)
    # dataset_normal = dgd.RomeDataset()
    model = dgd.DeepGD().to(device).float()

    if prev_model:
        model.load_state_dict(torch.load(prev_model))

    # do note that if you're switching devices you also need to switch the device of StressVP (this is done so that a new camera object is only created once)
    criteria = {
        dgd.StressVP(device) : 1,
        # dgd.Stress(): 1,
        # dgd.EdgeVar(): 0,
        # dgd.Occlusion(): 0,
        # dgd.IncidentAngle(): 0,
        # dgd.TSNEScore(): 0,
    }
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    datalist = list(dataset)
    random.seed(12345)
    random.shuffle(datalist)

    # for data in datalist:
    #     data.init_viewpoint = data.init_viewpoint.unsqueeze(1).T
    #     # data.x = data.x[:, 0:2]
    #     # data.init_pos = data.init_pos[:, 0:2]

    train_loader = pyg.loader.DataLoader(datalist[:10000], batch_size=batch_size, shuffle=True)
    val_loader = pyg.loader.DataLoader(datalist[11000:], batch_size=batch_size, shuffle=False)
    test_loader = pyg.loader.DataLoader(datalist[10000:11000], batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    print('starting training')

    for epoch in range(25):
        model.train()
        losses = []
        batch_cnt = 0
        for batch in tqdm(train_loader):
            batch = batch.to(device)
            model.zero_grad()
            loss = 0
            for c, w in criteria.items():
                loss += w * c(model(batch), batch)
            loss.backward()
            optim.step()
            losses.append(loss.item())
            # print(f'[Batch {batch_cnt}] Train Loss: {loss}')
            batch_cnt += 1
        print(f'[Epoch {epoch}] Train Loss: {np.mean(losses)}')
        train_losses.append(np.mean(losses))
        with torch.no_grad():
            model.eval()
            losses = []
            for batch in tqdm(val_loader, disable=True):
                batch = batch.to(device)
                loss = 0
                for c, w in criteria.items():
                    loss += w * c(model(batch), batch)
                losses.append(loss.item())
            print(f'[Epoch {epoch}] Val Loss: {np.mean(losses)}')
            val_losses.append(np.mean(losses))

    now = datetime.now()
    dt_string = now.strftime("%d-%m_%H-%M")

    # save the model
    model_name = dt_string + '_vwp_pred_stress'
    torch.save(model.state_dict(), 'saved_models/' + model_name + '.pt')

    # save the losses
    plt.plot(train_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Train loss')

    plt.title('Train loss progression for model: ' + model_name)
    plt.savefig('saved_models/' + model_name + '_train.png')
    plt.close('all')

    plt.plot(val_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Validation loss')

    plt.title('Validation loss progression for model: ' + model_name)
    plt.savefig('saved_models/' + model_name + '_validation.png')
    plt.close('all')
