import os
import json
import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import get_linear_schedule_with_warmup

from model import RetrievalModel
from recall import evaluation
import tqdm
import argparse
import gc

device = torch.device("cuda:0")


class LinkedFeatureDataset(Dataset):
    def __init__(self, links):
        self.links = links
        self.text_cache = {}
        self.pc_cache = {}

    def __len__(self):
        return len(self.links)

    def __getitem__(self, idx):
        text_info, pc_info = self.links[idx]
        text_path, text_idx = text_info
        pc_path, pc_idx = pc_info

        if text_path not in self.text_cache:
            self.text_cache[text_path] = torch.load(text_path, map_location='cpu')
        text_feats = self.text_cache[text_path]  # [num_texts, D]
        text_feat = text_feats[text_idx]  # [D]

        if pc_path not in self.pc_cache:
            self.pc_cache[pc_path] = torch.load(pc_path, map_location='cpu')
        pc_feats = self.pc_cache[pc_path]  # [num_pcs, D]
        pc_feat = pc_feats[pc_idx]  # [D]

        return text_feat, pc_feat, torch.tensor(pc_idx)


def collate_fn(batch):
    text_list, pc_list, pc_indices = [], [], []
    for text, pc, pc_idx in batch:
        text_list.append(text)
        pc_list.append(pc)
        pc_indices.append(pc_idx)

    text_feats = torch.stack(text_list, dim=0)  # [B, D]
    pc_feats = torch.stack(pc_list, dim=0)  # [B, D]
    pc_indices = torch.tensor(pc_indices)  # [B]

    return text_feats, pc_feats, pc_indices


def get_all_T3D(link, txt_pth, cld_pth):
    txt_features = torch.load(txt_pth)
    cld_features = torch.load(cld_pth)
    link = [[txt[1], cld[1]] for txt, cld in link]
    return txt_features, cld_features, link


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state loaded.")
        else:
            print("Scheduler state not found in checkpoint, starting scheduler from scratch.")
        start_epoch = checkpoint['epoch']
        best_recall_at_1 = checkpoint['best_recall_at_1']
        best_recall_at_5 = checkpoint['best_recall_at_5']
        best_recall_at_10 = checkpoint['best_recall_at_10']

        print(f"Checkpoint loaded, starting from epoch {start_epoch}")
        return start_epoch, best_recall_at_1, best_recall_at_5, best_recall_at_10
    else:
        print("No checkpoint found, starting from scratch")
        return 0, 0, 0, 0


def train_save_checkpoint(model, dataloader, optimizer, scheduler, txt, cld, link,
                          save_model_path, checkpoint_path, num_epochs=100):
    start_epoch, best_recall_at_1, best_recall_at_5, best_recall_at_10 = load_checkpoint(model, optimizer, scheduler,
                                                                                         checkpoint_path)
    txt, cld = txt.to(device), cld.to(device)

    for epoch in tqdm.trange(start_epoch, num_epochs):
        print("Epoch——————", epoch + 1, flush=True)
        model.train()
        epoch_losses = []
        for text, point_clouds, pc_indices in dataloader:
            text, point_clouds, pc_indices = text.to(device), point_clouds.to(device), pc_indices.to(device)
            optimizer.zero_grad()
            loss = model(text, point_clouds, pc_indices=pc_indices)
            print("batch_loss:", loss, flush=True)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(loss.item())

            del text, point_clouds, loss
            torch.cuda.empty_cache()
            gc.collect()

        epoch_loss = np.mean(epoch_losses)
        # current_lr = scheduler.get_last_lr()[0]
        # print(f"Epoch: {epoch + 1}, Current LR: {current_lr:.2e}", flush=True)

        model.eval()
        recalls, mr, mrr = evaluation(txt, cld, link, sim_func=model, t2c=True, recall_idx=[1, 5, 10])
        current_recall_at_1 = recalls[0]
        current_recall_at_5 = recalls[1]
        current_recall_at_10 = recalls[2]
        print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f},"
              f"Recall@1: {recalls[0]:.4f},  Recall@5: {recalls[1]:.4f}, Recall@10: {recalls[2]:.4f}", flush=True)

        base_path, ext = os.path.splitext(save_model_path)
        if current_recall_at_1 > best_recall_at_1:
            best_recall_at_1 = current_recall_at_1
            best_r1_path = f"{base_path}_r1{ext}"
            torch.save(model.state_dict(), best_r1_path)
        if current_recall_at_5 > best_recall_at_5:
            best_recall_at_5 = current_recall_at_5
            best_r5_path = f"{base_path}_r5{ext}"
            torch.save(model.state_dict(), best_r5_path)
        if current_recall_at_10 > best_recall_at_10:
            best_recall_at_10 = current_recall_at_10
            best_r10_path = f"{base_path}_r10{ext}"
            torch.save(model.state_dict(), best_r10_path)

        if (epoch + 1) % 25 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_recall_at_1': best_recall_at_1,
                'best_recall_at_5': best_recall_at_5,
                'best_recall_at_10': best_recall_at_10
            }
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch + 1}")

            del checkpoint
            torch.cuda.empty_cache()
            gc.collect()

        del recalls, mr, mrr
        torch.cuda.empty_cache()
        gc.collect()

    print(f"Final best R@1: {best_recall_at_1}")
    print(f"Final best R@5: {best_recall_at_5}")
    print(f"Final best R@10: {best_recall_at_10}")


def parse_command_line():
    parser = argparse.ArgumentParser(description='Train a model and save parameters.')

    parser.add_argument('--save_model_path', "-s", default="./best_model/bestEle.pth",
                        type=str, help='The path to save model parameters')
    parser.add_argument('--checkpoint_path', "-c", default="./checkpoint/Ele.pth",
                        type=str, help='The path to save checkpoint')
    parser.add_argument('--train_txt_path', "-trtp",
                        default="./feature/Elephant/train/txt_features_CLIP.pt",
                        type=str, help='train txt feature path')
    parser.add_argument('--train_cld_path', "-trcp",
                        default="./feature/Elephant/train/pc_features_DGCNN.pt",
                        type=str, help='train cld feature path')
    parser.add_argument('--train_link_path', "-trl", default="./save/Elephant_link_train.json",
                        type=str, help='train link file path')

    parser.add_argument('--test_txt_path', "-tetp", default="./feature/Elephant/test/txt_features_CLIP.pt",
                        type=str, help='test txt feature path')
    parser.add_argument('--test_cld_path', "-tecp", default="./feature/Elephant/test/pc_features_DGCNN.pt",
                        type=str, help='test cld feature path')
    parser.add_argument('--test_link_path', "-tel", default="./save/Elephant_link_test.json",
                        type=str, help='test link file path')

    parser.add_argument('--batch_size', "-b", default=256,
                        type=int, help='batch size')
    parser.add_argument('--nhead', "-nh", default=64,
                        type=int, help='attention head numbers')
    parser.add_argument('--dropout', "-dp", default=0.1,
                        type=float, help='dropout rate')
    parser.add_argument('--num_layers', "-layers", default=6,
                        type=int, help='attention encoder numbers')
    parser.add_argument('--hyperbolic_c', type=float, default=1.0,
                        help='Parameter c for the hyperbolic curvature, which is defined as -c. ' \
                             'Default c=1.0 results in a curvature of -1, representing the standard hyperbolic space.')
    parser.add_argument('--loss_temp', type=float, default=0.07,
                        help='Temperature for contrastive loss')
    parser.add_argument('--loss_lambda', type=float, default=0.2,
                        help='Weight for the entailment loss component')
    parser.add_argument('--num_epochs', "-epo", default=100,
                        type=int, help='epoch numbers')
    parser.add_argument('--learning_rate', "-lr", default=0.002,
                        type=float, help='learning rate')
    parser.add_argument('--warmup_ratio', default=0.1,
                        type=float, help='Ratio of total training steps for learning rate warmup')
    parser.add_argument('--rank', "-rk", default=256,
                        type=int, help='rank')
    parser.add_argument('--d_txt', "-dtxt", default=512,
                        type=int, help='d_txt')
    parser.add_argument('--d_pcd', "-dpcd", default=512,
                        type=int, help='d_pcd')
    parser.add_argument('--d_model', "-dm", default=512,
                        type=int, help='d_model')

    args = parser.parse_args()
    return args


def t_Lmodel():
    args = parse_command_line()

    with open(args.train_link_path, "r", encoding='utf-8') as f:
        train_link = json.load(f)
    txt_train_size = train_link[-1][0][1]
    cld_train_size = train_link[-1][1][1]
    print(f'train txt size: {txt_train_size + 1}\ncld size: {cld_train_size + 1}\nlink size: {len(train_link)}')
    train_dataset = LinkedFeatureDataset(train_link)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True
    )
    with open(args.test_link_path, "r", encoding='utf-8') as f:
        test_link = json.load(f)
    txt_test, cld_test, link_test = get_all_T3D(test_link, args.test_txt_path, args.test_cld_path)
    print(f'test txt size: {txt_test.size()}\ncld size: {cld_test.size()}\nlink size: {len(link_test)}')

    model = RetrievalModel(
        d_model=args.d_model,
        d_txt=args.d_txt,
        d_pcd=args.d_pcd,
        n_head=args.nhead,
        dropout=args.dropout,
        num_layers=args.num_layers,
        hyperbolic_c=args.hyperbolic_c,
        loss_temp=args.loss_temp,
        loss_lambda=args.loss_lambda
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.91, 0.9993), eps=1e-8)

    num_total_steps = len(train_loader) * args.num_epochs
    num_warmup_steps = int(num_total_steps * args.warmup_ratio)
    print(f"Total training steps: {num_total_steps}, Warmup steps: {num_warmup_steps}", flush=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_total_steps
    )

    train_save_checkpoint(model, train_loader, optimizer, scheduler, txt_test, cld_test, link_test,
                          args.save_model_path, args.checkpoint_path, args.num_epochs)


if __name__ == '__main__':

    t_Lmodel()
