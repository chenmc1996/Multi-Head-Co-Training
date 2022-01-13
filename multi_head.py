import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler
import os


def ce_loss(logits, targets, reduction='none'):
    return F.cross_entropy(logits, targets, reduction=reduction)

def consistency_loss_tri(logits_x_ulbs):
    logits_x_ulb_ws = [i[0] for i in logits_x_ulbs]
    logits_x_ulb_ss = [i[1] for i in logits_x_ulbs]
    N = logits_x_ulb_ws[0].shape[0]
    d = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    loss = []
    for i in d:
        _, max_idx_1 = torch.max(logits_x_ulb_ws[d[i][0]], dim=-1)
        _, max_idx_2 = torch.max(logits_x_ulb_ws[d[i][1]], dim=-1)
        mask = (max_idx_1 == max_idx_2)
        loss.append(
            ce_loss(logits_x_ulb_ss[i][mask], max_idx_1[mask], reduction='sum')/N)
    return loss


class Multi_Head:
    def __init__(self, net_builder, num_classes, ema_m,  lambda_u, it=0, num_eval_iter=1000):

        super(Multi_Head, self).__init__()

        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        self.train_model = net_builder(num_classes=num_classes)
        self.eval_model = net_builder(num_classes=num_classes)

        self.num_eval_iter = num_eval_iter
        self.lambda_u = lambda_u

        self.optimizer = None
        self.scheduler = None

        self.it = 0

        for param_q, param_k in zip(self.train_model.parameters(), self.eval_model.parameters()):
            param_k.data.copy_(param_q.detach().data)
            param_k.requires_grad = False

        self.eval_model.eval()

    @torch.no_grad()
    def _eval_model_update(self):
        for param_train, param_eval in zip(self.train_model.parameters(), self.eval_model.parameters()):
            param_eval.copy_(param_eval * self.ema_m +
                             param_train.detach() * (1-self.ema_m))

        for buffer_train, buffer_eval in zip(self.train_model.buffers(), self.eval_model.buffers()):
            buffer_eval.copy_(buffer_train)

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        print(f'[!] data loader keys: {self.loader_dict.keys()}')

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, args):

        self.train_model.train()

        best_eval_acc, best_it = 0.0, 0

        scaler = GradScaler()
        amp_cm = autocast

        for (x_lb, y_lb), (x_ulb_w, x_ulb_s, _) in zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb']):
            if self.it > args.num_train_iter:
                break

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]

            x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), [
                i.cuda(args.gpu) for i in x_ulb_s]
            y_lb = y_lb.cuda(args.gpu)

            inputs = torch.cat((x_lb, x_ulb_w, *x_ulb_s))

            with amp_cm():

                logits = self.train_model(inputs, share_num=num_lb+num_ulb)
                logits_x_lbs = [i[:num_lb] for i in logits]
                logits_x_ulb_s = [i[num_lb:].chunk(2) for i in logits]

                del logits

                sup_loss = [ce_loss(i, y_lb, reduction='mean')
                            for i in logits_x_lbs]

                sup_loss = sum(sup_loss)

                unsup_loss = consistency_loss_tri(logits_x_ulb_s)

                unsup_loss = sum(unsup_loss)

                total_loss = sup_loss + self.lambda_u * unsup_loss

            scaler.scale(total_loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            self.scheduler.step()
            self.train_model.zero_grad()

            with torch.no_grad():
                self._eval_model_update()

            if self.it % self.num_eval_iter == 0:

                eval_dict = self.evaluate(args=args)

                save_path = os.path.join(args.save_dir, args.save_name)

                print(f"{self.it} iteration, EVAL_ACC: {eval_dict['acc']}")

                self.save_model('model_latest.pth', save_path)

            self.it += 1

            if self.it > 2**19:
                self.num_eval_iter = 1000

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None, EMA=True):
        if EMA:
            eval_model = self.eval_model
        else:
            eval_model = self.train_model

        eval_model.eval()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        result = []
        all_y = []

        total_loss = 0.0
        total_acc = 0.0
        total_num = 0.0
        for x, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits = sum(eval_model(x, 0))
            loss = F.cross_entropy(logits, y, reduction='mean')
            acc = torch.sum(torch.max(logits, dim=-1)[1] == y)

            total_loss += loss.detach()*num_batch
            total_acc += acc.detach()
        if EMA:
            pass
        else:
            eval_model.train()

        return {'loss': total_loss/total_num, 'acc': total_acc/total_num}

    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        train_model = self.train_model.module if hasattr(
            self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(
            self.eval_model, 'module') else self.eval_model
        torch.save({'train_model': train_model.state_dict(),
                    'eval_model': eval_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it}, save_filename)

        print(f"model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)

        train_model = self.train_model.module if hasattr(
            self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(
            self.eval_model, 'module') else self.eval_model
        new_head = True

        for key in checkpoint.keys():
            if hasattr(self, key) and getattr(self, key) is not None:
                if 'train_model' in key:
                    train_model.load_state_dict(checkpoint[key])
                elif 'eval_model' in key:
                    eval_model.load_state_dict(checkpoint[key])
                elif key == 'scheduler':
                    self.scheduler.load_state_dict(checkpoint[key])
                elif key == 'optimizer':
                    self.optimizer.load_state_dict(checkpoint[key])
                elif key == 'it':
                    self.it = checkpoint[key]
                else:
                    getattr(self, key).load_state_dict(checkpoint[key])
                print(f"Check Point Loading: {key} is LOADED")
            else:
                print(f"Check Point Loading: {key} is **NOT** LOADED")


if __name__ == "__main__":
    pass
