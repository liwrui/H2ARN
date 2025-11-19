import os
import json
import torch
import argparse

from model import RetrievalModel
from recall import evaluation

device = torch.device("cuda:0")


def get_all_T3D(link, txt_pth, cld_pth):
    txt_features = torch.load(txt_pth)
    cld_features = torch.load(cld_pth)
    link = [[txt[1], cld[1]] for txt, cld in link]
    return txt_features, cld_features, link


def load_best_model(model, best_model_path):
    if os.path.exists(best_model_path):
        best_model_wts = torch.load(best_model_path)
        model.load_state_dict(best_model_wts)
        print(f"Best model loaded from {best_model_path}")
    else:
        print(f"Not found {best_model_path}, unable to load model.")
    return model


def parse_command_line():
    parser = argparse.ArgumentParser(description='Train a model and save parameters.')

    parser.add_argument('--best_model_path', "-s", default="./best_model/bestEle.pth",
                        type=str, help='The path to save model parameters')
    parser.add_argument('--test_txt_path', "-tetp", default="./feature/Elephant/test/txt_features_CLIP.pt",
                        type=str, help='test txt feature path')
    parser.add_argument('--test_cld_path', "-tecp", default="./feature/Elephant/test/pc_features_DGCNN.pt",
                        type=str, help='test cld feature path')
    parser.add_argument('--test_link_path', "-tel", default="./save/Elephant_link_test.json",
                        type=str, help='test link file path')

    parser.add_argument('--d_model', "-dm", default=512,
                        type=int, help='d_model')
    parser.add_argument('--d_txt', "-dtxt", default=512,
                        type=int, help='d_txt')
    parser.add_argument('--d_pcd', "-dpcd", default=512,
                        type=int, help='d_pcd')
    parser.add_argument('--nhead', "-nh", default=64,
                        type=int, help='attention head numbers')
    parser.add_argument('--dropout', "-dp", default=0.1,
                        type=float, help='dropout rate')
    parser.add_argument('--rank', "-rk", default=256,
                        type=int, help='rank')
    parser.add_argument('--num_layers', "-layers", default=6,
                        type=int, help='attention encoder numbers')
    parser.add_argument('--hyperbolic_c', type=float, default=1.0,
                        help='Parameter c for the hyperbolic curvature, which is defined as -c. ' \
                             'Default c=1.0 results in a curvature of -1, representing the standard hyperbolic space.')
    parser.add_argument('--loss_temp', type=float, default=0.07,
                        help='Temperature for contrastive loss')
    parser.add_argument('--loss_lambda', type=float, default=0.2,
                        help='Weight for the entailment loss component')

    args = parser.parse_args()
    return args


def inference():
    args = parse_command_line()

    with open(args.test_link_path, "r", encoding='utf-8') as f:
        test_link = json.load(f)
    txt_test, cld_test, link_test = get_all_T3D(test_link, args.test_txt_path, args.test_cld_path)
    print(f'test txt size: {txt_test.size()}\ncld size: {cld_test.size()}\nlink size: {len(link_test)}')
    
    txt_test, cld_test = txt_test.to(device), cld_test.to(device)

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
    model = load_best_model(model, args.best_model_path)

    model.eval()

    with torch.no_grad():
        print("\n--- Starting Evaluation: Text-to-Cloud ---")
        recalls_t2c, mr_t2c, mrr_t2c = evaluation(txt_test, cld_test, link_test, sim_func=model, t2c=True, recall_idx=[1, 5, 10])
        print(f"T2P Results -> R@1: {recalls_t2c[0]:.4f}, R@5: {recalls_t2c[1]:.4f}, R@10: {recalls_t2c[2]:.4f}")
        print(f"              MR: {mr_t2c:.2f}, MRR: {mrr_t2c:.4f}")

        print("\n--- Starting Evaluation: Cloud-to-Text ---")
        recalls_c2t, mr_c2t, mrr_c2t = evaluation(txt_test, cld_test, link_test, sim_func=model, t2c=False, recall_idx=[1, 5, 10])
        print(f"P2T Results -> R@1: {recalls_c2t[0]:.4f}, R@5: {recalls_c2t[1]:.4f}, R@10: {recalls_c2t[2]:.4f}")
        print(f"              MR: {mr_c2t:.2f}, MRR: {mrr_c2t:.4f}")
    
    print("\n--- All evaluations completed ---")

if __name__ == '__main__':

    inference()
