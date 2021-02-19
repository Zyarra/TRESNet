import torch
import torch.nn.functional as F
import time
import numpy as np

scaler = torch.cuda.amp.GradScaler()


def train(model, optimizer, data_loader, loss_history, use_cuda=True, use_fp16=True):
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    model.train()
    i = 0
    avg_step_time = []
    for data, target in data_loader:
        step_start = time.time()
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        if use_fp16:
            with torch.cuda.amp.autocast():
                output = F.log_softmax(model(data), dim=1)
                loss = F.nll_loss(output, target)
            scaler.scale(loss).backward()
            _, pred = torch.max(output, dim=1)
            correct_samples += pred.eq(target).sum()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target)
            loss.backward()
            _, pred = torch.max(output, dim=1)
            correct_samples += pred.eq(target).sum()
            optimizer.step()
        loss_history.append(loss.item())
        if i > 0:
            print(
                f'\r[{(i * len(data)):5}/{total_samples:5}, ({(100 * i / len(data_loader)):3.0f}% )] Loss: [{np.mean(loss_history[-i:]):6.4f}], Training Accuracy: [{(100.0 * correct_samples / total_samples):4.2f}%], Time left: {np.mean(avg_step_time):5.2f} seconds',
                end='')
        i += 1
        avg_step_time.append((time.time() - step_start) * (len(data_loader) - i))


def test(model, data_loader, loss_history, use_cuda=True):
    model.eval()
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0
    i = 0
    with torch.no_grad():
        for data, target in data_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()
            print(
                f'\rTesting [{((i / len(data_loader)) * 100):3.2f}%] Test loss: [{(total_loss / total_samples):6.4f}], Test accuracy: {correct_samples:5} / {total_samples:5} ({(100.0 * correct_samples / total_samples):4.2f}%)',
                end='')
            i += 1
    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print(
        f'\rAverage test loss:[{avg_loss:.4f}], Test Accuracy:[{correct_samples}/{total_samples}] ({(100.0 * correct_samples / total_samples):4.2f}%)')
