import argparse
import json
import os
import torch
import open3d as o3d
import numpy as np
import gc
from collections import OrderedDict

from backbone.pc_encoder.PointNet.PointNet import PointNetEncoder
from backbone.pc_encoder.PointNet2.PointNet2 import PointNet2Encoder
from backbone.pc_encoder.DGCNN.DGCNN import DGCNN
from backbone.txt_encoder.CLIP_encoder import encode_text_single
from backbone.txt_encoder.Bert_encoder import encode_text_single_with_bert
from transformers import CLIPProcessor, CLIPModel, BertModel, BertTokenizer


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def augment_pointcloud_to_9d(pointcloud_6d):
    xyz = pointcloud_6d[:, :3]

    xyz_min = np.min(xyz, axis=0)
    xyz_max = np.max(xyz, axis=0)
    xyz_range = xyz_max - xyz_min
    xyz_range[xyz_range == 0] = 1e-8

    normalized_xyz = (xyz - xyz_min) / xyz_range

    pointcloud_9d = np.hstack((pointcloud_6d[:, :6], normalized_xyz))
    
    return pointcloud_9d


def get_t3d_link(args):

    print(f"\n{'='*25}")
    print(f"Processing split: {args.split.upper()}")
    print(f"{'='*25}")

    try:
        with open(args.caption_json, 'r', encoding='utf-8') as f:
            captions = json.load(f)
        print(f"Successfully loaded caption json: {args.caption_json}", flush=True)
    except Exception as e:
        print(f"Error loading caption JSON {args.caption_json}: {e}", flush=True)
        return

    print("PC Model--------------", args.pc_model)
    print("Txt Model-------------", args.txt_model)

    pc_model_instance = None
    pc_models_by_area = {}
    try:
        if args.pc_model == 'pointnet':
            pc_model_instance = PointNetEncoder(args).to(args.device)
            state_dict = torch.load(args.save_PointNet, map_location=args.device)
            pc_model_instance.load_state_dict(state_dict, strict=False)
            print("Successfully loaded PC encoder model PointNet!", flush=True)
            pc_model_instance.eval()
        elif args.pc_model == 'pointnet++':
            pc_model_instance = PointNet2Encoder(args).to(args.device)
            state_dict = torch.load(args.save_PointNet2, map_location=args.device)
            pc_model_instance.load_state_dict(state_dict, strict=False)
            print("Successfully loaded PC encoder model PointNet++!", flush=True)
            pc_model_instance.eval()
        elif args.pc_model == 'DGCNN':
            print("Pre-loading DGCNN models for all areas (1-6)...")
            for i in range(1, 7):
                area_str = str(i)
                model_path = os.path.join(args.save_DGCNN_root, f'model_{area_str}.t7')
                if not os.path.exists(model_path):
                    print(f"Warning: Model for Area {area_str} not found at {model_path}. Skipping.")
                    continue

                pc_model_instance = DGCNN(args).to(args.device)
                state_dict = torch.load(model_path, map_location=args.device)
                
                if all(k.startswith('module.') for k in state_dict.keys()):
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]
                        new_state_dict[name] = v
                    pc_model_instance.load_state_dict(new_state_dict, strict=False)
                else:
                    pc_model_instance.load_state_dict(state_dict, strict=False)
                
                pc_model_instance.eval()
                pc_models_by_area[area_str] = pc_model_instance
                print(f"Successfully loaded model for Area {area_str}.")
        else:
            raise Exception(f"PC model '{args.pc_model}' not implemented or invalid.")
    except Exception as e:
        print(f"Error loading PC encoder model: {e}", flush=True)
        return
    
    text_model = None
    text_processor = None
    text_tokenizer = None
    try:
        if args.txt_model == 'CLIP':
            print("Loading CLIP model and processor once...", flush=True)
            local_dir = args.save_CLIP
            text_model = CLIPModel.from_pretrained(local_dir).to(args.device)
            text_processor = CLIPProcessor.from_pretrained(local_dir)
            text_model.eval()
            print("CLIP model loaded successfully.", flush=True)
        elif args.txt_model == 'BERT':
            local_dir = args.save_BERT
            text_model = BertModel.from_pretrained(local_dir).to(args.device)
            text_tokenizer = BertTokenizer.from_pretrained(local_dir)
            text_model.eval()
            print("BERT model loaded successfully.", flush=True)
    except Exception as e:
        print(f"Error loading Text encoder model: {e}", flush=True)
        return
    
    all_pc_features_list = []
    all_txt_features_list = []
    link_data = []
    pc_name_to_idx = {}

    for obj_name, obj_captions in captions["result"].items():
        print(f"--- Processing scene: {obj_name} ---", flush=True)
        pc_model_instance = None
        try:
            area_num = obj_name.split('_')[1]
            pc_model_instance = pc_models_by_area.get(area_num)
            
            if pc_model_instance is None:
                print(f"Warning: No pre-loaded model available for Area {area_num}. Skipping scene {obj_name}.")
                continue
        except (IndexError, KeyError):
            print(f"Warning: Could not determine area from scene name '{obj_name}'. Skipping.")
            continue

        current_pc_idx_in_list = pc_name_to_idx.get(obj_name, -1)

        if current_pc_idx_in_list == -1:
            pc_path = os.path.join(args.pcd_dir, f"{obj_name}.txt")
            if not os.path.exists(pc_path):
                print(f"Warning: Point cloud file not found: {pc_path}. Skipping scene {obj_name}.", flush=True)
                continue
            
            try:
                points = np.loadtxt(pc_path, delimiter=' ')
                if points.shape[1] < 6:
                    print(f"Error: Point cloud file {pc_path} has fewer than 6 columns. Skipping.")
                    continue
                pc_data = augment_pointcloud_to_9d(points)
                
                pc_tensor_on_device = torch.tensor(pc_data, dtype=torch.float32).unsqueeze(0).transpose(1, 2).contiguous().to(args.device)
                
                print(f"PC tensor shape for {obj_name}: {pc_tensor_on_device.shape}", flush=True)
                with torch.no_grad():
                    pc_feature_tensor_on_device = pc_model_instance(pc_tensor_on_device).detach()

                    if args.pc_model in ['pointnet++', 'DGCNN']:
                        pc_feature_tensor_on_device = pc_feature_tensor_on_device.transpose(1, 2)
                    print(f"PC feature shape for {obj_name}: {pc_feature_tensor_on_device.shape}", flush=True)

                all_pc_features_list.append(pc_feature_tensor_on_device.cpu().squeeze(0))
                current_pc_idx_in_list = len(all_pc_features_list) - 1
                pc_name_to_idx[obj_name] = current_pc_idx_in_list
                
                del pc_tensor_on_device, pc_feature_tensor_on_device, points, pc_data
                clear_gpu_cache()

            except Exception as e:
                print(f"Error processing point cloud {obj_name}: {e}. Skipping related captions.", flush=True)
                current_pc_idx_in_list = -1

        else:
            current_pc_idx_in_list = pc_name_to_idx[obj_name]
            print(f"PC feature for {obj_name} already processed at index {current_pc_idx_in_list}.", flush=True)
        
        if current_pc_idx_in_list != -1:
            for i, txt_caption in enumerate(obj_captions):
                current_txt_idx_in_list = len(all_txt_features_list)
                
                print(f"Processing caption {i+1} for {obj_name} (Global Txt Index: {current_txt_idx_in_list})", flush=True)
                
                try:
                    txt_feature_tensor_on_device = None
                    if args.txt_model == 'CLIP':
                        txt_feature_tensor_on_device = encode_text_single(
                            txt_caption,
                            text_model,     
                            text_processor, 
                            args.device
                        ).detach()

                    elif args.txt_model == 'BERT':
                        txt_feature_tensor_on_device = encode_text_single_with_bert(
                            txt_caption,
                            text_model,     
                            text_tokenizer, 
                            args.device
                        ).detach()
                    else:
                        raise Exception(f"Text model '{args.txt_model}' not implemented or invalid.")
                    print(f"Txt feature shape: {txt_feature_tensor_on_device.shape}", flush=True)

                    all_txt_features_list.append(txt_feature_tensor_on_device.cpu().squeeze(0))

                    link_data.append([
                        [args.txt_feature_path, current_txt_idx_in_list],
                        [args.pc_feature_path, current_pc_idx_in_list]
                    ])
                    
                    del txt_feature_tensor_on_device
                    clear_gpu_cache()

                except Exception as e:
                    print(f"Error processing caption {i+1} for {obj_name}: {e}. Skipping this caption.", flush=True)

    print("\n--- Saving all accumulated features... ---", flush=True)
    
    os.makedirs(os.path.dirname(args.txt_feature_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.pc_feature_path), exist_ok=True)

    if all_pc_features_list:
        final_pc_features_tensor = torch.stack(all_pc_features_list)
        torch.save(final_pc_features_tensor, args.pc_feature_path)
        print(f"All PC features for {args.split} saved to {args.pc_feature_path} with shape {final_pc_features_tensor.shape}", flush=True)
    else:
        print("No PC features to save.", flush=True)

    if all_txt_features_list:
        final_txt_features_tensor = torch.stack(all_txt_features_list)
        torch.save(final_txt_features_tensor, args.txt_feature_path)
        print(f"All Txt features for {args.split} saved to {args.txt_feature_path} with shape {final_txt_features_tensor.shape}", flush=True)
    else:
        print("No Txt features to save.", flush=True)

    with open(args.link_json, 'w', encoding='utf-8') as f:
        json.dump(link_data, f, indent=4, ensure_ascii=False)

    print(f"--- Successfully saved features and link data for {args.split.upper()} split! ---", flush=True)
    print(f"Total PC features processed: {len(all_pc_features_list)}")
    print(f"Total Txt features processed: {len(all_txt_features_list)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract features from S3DIS dataset")

    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., cuda:0 or cpu)')
    parser.add_argument('--base_data_dir', type=str, help='Base directory of the original dataset.')
    parser.add_argument('--base_output_dir', type=str, default='./feature/S3DIS', help='Base directory to save output features.')
    parser.add_argument('--link_save_dir', type=str, default='./save/', help='Directory to save the final link.json file.')

    parser.add_argument('--pc_model', type=str, default='DGCNN', choices=['pointnet', 'pointnet++', 'DGCNN'], help='Point cloud encoder model to use.')
    parser.add_argument('--txt_model', type=str, default='CLIP', choices=['CLIP', 'BERT'], help='Text encoder model to use.')

    parser.add_argument('--save_PointNet', type=str, default='./save/PointNet/PointNet_best_model.pth', help='Path to pre-trained PointNet model weights.')
    parser.add_argument('--save_PointNet2', type=str, default='./save/PointNet2/PointNet2_best_model.pth', help='Path to pre-trained PointNet++ model weights.')
    parser.add_argument('--save_DGCNN_root', type=str, default='./save/DGCNN', help='Root directory of area-specific DGCNN models.')
    parser.add_argument('--save_CLIP', type=str, default='./save/CLIP', help='Path to the directory containing the saved CLIP model and processor files.')
    parser.add_argument('--save_BERT', type=str, default='./save/BERT', help='Path to the directory containing the saved BERT model and tokenizer files.')

    parser.add_argument('--input_dim', type=int, default=3, help='Input dimension for point cloud (e.g., 3 for XYZ).') # PointNet
    parser.add_argument('--output_dim', type=int, default=512, help='Output dimension for PointNet features.') # PointNet
    parser.add_argument('--feature_transform', type=bool, default=False, help='Whether to use feature transform in PointNet.') # PointNet
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings for DGCNN') # DGCNN
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use for DGCNN') # DGCNN
    
    args = parser.parse_args()

    # --- Loop through train and test splits ---
    for split in ['train', 'test']:
        args.split = split

        args.caption_json = os.path.join(args.base_data_dir, f'S3DIS_{split}_captions.json')
        args.pcd_dir = os.path.join(args.base_data_dir, f'S3DIS_{split}_pcds')
        
        output_dir = os.path.join(args.base_output_dir, split)
        args.txt_feature_path = os.path.join(output_dir, f'txt_features_{args.txt_model}.pt')
        args.pc_feature_path = os.path.join(output_dir, f'pc_features_{args.pc_model}.pt')

        os.makedirs(args.link_save_dir, exist_ok=True) 
        link_filename = f'S3DIS_link_{split}.json'
        args.link_json = os.path.join(args.link_save_dir, link_filename)

        get_t3d_link(args)
