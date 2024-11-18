import sys
import os
from trimesh import PointCloud
sys.path.append(os.getcwd())
from glob import glob
import gen_utils as gu
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import copy
import argparse

os.environ["DISPLAY"] = ""  # Disables GUI rendering
os.environ["OPEN3D_CPU_RENDERING"] = "true"

parser = argparse.ArgumentParser(description='Inference models')
parser.add_argument('--mesh_path', default="G:/tooth_seg/main/all_datas/chl/3D_scans_per_patient_obj_files/013FHA7K/013FHA7K_lower.obj", type=str)
parser.add_argument('--gt_json_path', default="G:/tooth_seg/main/all_datas/chl/ground-truth_labels_instances/013FHA7K/013FHA7K_lower.json" ,type=str)
parser.add_argument('--pred_json_path', type=str, default="test_results/013FHA7K_lower.json")
args = parser.parse_args()


def cal_metric(gt_labels, pred_sem_labels, pred_ins_labels, is_half=None, vertices=None):
    ins_label_names = np.unique(pred_ins_labels)
    ins_label_names = ins_label_names[ins_label_names != 0]
    IOU = 0
    F1 = 0
    ACC = 0
    SEM_ACC = 0
    IOU_arr = []
    for ins_label_name in ins_label_names:
        #instance iou
        ins_label_name = int(ins_label_name)
        ins_mask = pred_ins_labels==ins_label_name
        gt_label_uniqs, gt_label_counts = np.unique(gt_labels[ins_mask], return_counts=True)
        gt_label_name = gt_label_uniqs[np.argmax(gt_label_counts)]
        gt_mask = gt_labels == gt_label_name

        TP = np.count_nonzero(gt_mask * ins_mask)
        FN = np.count_nonzero(gt_mask * np.invert(ins_mask))
        FP = np.count_nonzero(np.invert(gt_mask) * ins_mask)
        TN = np.count_nonzero(np.invert(gt_mask) * np.invert(ins_mask))

        ACC += (TP + TN) / (FP + TP + FN + TN)
        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        F1 += 2*(precision*recall) / (precision + recall)
        IOU += TP / (FP+TP+FN)
        IOU_arr.append(TP / (FP+TP+FN))
        #segmentation accuracy
        pred_sem_label_uniqs, pred_sem_label_counts = np.unique(pred_sem_labels[ins_mask], return_counts=True)
        sem_label_name = pred_sem_label_uniqs[np.argmax(pred_sem_label_counts)]
        if is_half:
            if sem_label_name == gt_label_name or sem_label_name + 8 == gt_label_name:
                SEM_ACC +=1
        else:
            if sem_label_name == gt_label_name:
                SEM_ACC +=1
        #print("gt is", gt_label_name, "pred is", sem_label_name, sem_label_name == gt_label_name)
    return IOU/len(ins_label_names), F1/len(ins_label_names), ACC/len(ins_label_names), SEM_ACC/len(ins_label_names), IOU_arr

gt_loaded_json = gu.load_json(args.gt_json_path)
gt_labels = np.array(gt_loaded_json['labels']).reshape(-1)

pred_loaded_json = gu.load_json(args.pred_json_path)
pred_labels = np.array(pred_loaded_json['labels']).reshape(-1)

IoU, F1, Acc, SEM_ACC, _ = cal_metric(gt_labels, pred_labels, pred_labels) # F1 -> TSA, SEM_ACC -> TIR
print("IoU", IoU, "F1(TSA)", F1, "SEM_ACC(TIR)", SEM_ACC)
_, mesh = gu.read_txt_obj_ls(args.mesh_path, ret_mesh=True, use_tri_mesh=True)
#gu.print_3d(gu.get_colored_mesh(mesh, gt_labels), save_as_image=True, image_path="gt_output.png") # color is random
#gu.print_3d(gu.get_colored_mesh(mesh, pred_labels)) # color is random
#gu.print_3d(gu.get_colored_mesh(mesh, gt_labels), save_as_image=True, image_path="./gt_output.png")
def visualize_results(mesh, labels, output_path):
    try:
        # Create colored mesh first
        colored_mesh = gu.get_colored_mesh(mesh, labels)
        
        # Save the colored mesh as PLY
        ply_path = output_path.replace('.png', '.ply')
        o3d.io.write_triangle_mesh(ply_path, colored_mesh)
        print(f"Colored mesh saved as {ply_path}")
        
        # Generate a simple 2D projection using numpy
        vertices = np.asarray(colored_mesh.vertices)
        colors = np.asarray(colored_mesh.vertex_colors)
        
        # Project 3D points to 2D
        # Using orthographic projection
        resolution = 1024
        img = np.ones((resolution, resolution, 3), dtype=np.uint8) * 255  # White background
        
        if len(vertices) > 0:
            # Normalize coordinates to fit in image
            min_vals = np.min(vertices, axis=0)
            max_vals = np.max(vertices, axis=0)
            
            # Scale points to image space
            scaled_points = (vertices - min_vals) / (max_vals - min_vals)
            scaled_points[:, [0, 1]] *= (resolution - 1)
            scaled_points = scaled_points.astype(int)
            
            # Sort points by depth (z coordinate) to handle occlusion
            depth_order = np.argsort(vertices[:, 2])
            
            # Draw points
            for idx in depth_order:
                x, y = scaled_points[idx, 0], scaled_points[idx, 1]
                if 0 <= x < resolution and 0 <= y < resolution:
                    # Convert color to uint8
                    color = (colors[idx] * 255).astype(np.uint8)
                    # Draw a small circle around each point
                    radius = 2
                    for dx in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            if dx*dx + dy*dy <= radius*radius:
                                new_x, new_y = x + dx, y + dy
                                if 0 <= new_x < resolution and 0 <= new_y < resolution:
                                    img[new_y, new_x] = color
        
        # Save the image
        from PIL import Image
        im = Image.fromarray(img)
        im.save(output_path)
        print(f"2D projection saved as {output_path}")
        
    except Exception as e:
        print(f"Visualization failed: {str(e)}")
        # Save raw vertex and color data as backup
        try:
            np.savez(output_path.replace('.png', '_backup.npz'), 
                    vertices=np.asarray(mesh.vertices),
                    colors=np.asarray(mesh.vertex_colors))
            print(f"Saved raw data as {output_path.replace('.png', '_backup.npz')}")
        except Exception as e2:
            print(f"Failed to save backup data: {str(e2)}")

# Replace your final visualization calls with:
_, mesh = gu.read_txt_obj_ls(args.mesh_path, ret_mesh=True, use_tri_mesh=True)
visualize_results(mesh, gt_labels, "./gt_output.png")
visualize_results(mesh, pred_labels, "./pred_output.png")