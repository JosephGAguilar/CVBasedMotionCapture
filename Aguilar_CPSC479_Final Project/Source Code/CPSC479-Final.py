#This is adapted from the demo.py script in https://github.com/mks0601/3DMPPE_POSENET_RELEASE
#and all of its dependencies from the repo. I modified this script to synthesize multiple parts from the
#RootNet repo, PoseNet repo ,as well as code from the "You only look once: Unified, real-time object detection"
#paper into functions that work in succession. The bouding box scripts (consolidate_bounding_boxes, bbox_extractor)
#are original from. extract_root_depth, extract_pose are from demo.py in the PoseNet paper, which
#are modified to utilize CPU instead of GPU. process_video is also an original function to handle video input.
#In summary, I reused code from the demo.py script of the PoseNet paper, reformatting and refunctioning
#the code to work in tandem with other neural networks (RootNet, YOLO). In addition, I retooled the entire repository
#to work with CPU instead of GPU. I also utilized the processes in an automatic pipleine to work on video input.


import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import math
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
import time

sys.path.append('main/')
from PoseConfig import p_cfg
from RootConfig import r_cfg
from PoseModel import get_pose_net
from RootModel import get_root_net
from PoseDataset import p_generate_patch_image
from RootDataset import r_generate_patch_image
sys.path.append('common/')
from utils.PoseUtils import p_process_bbox, p_pixel2cam
from utils.PoseVis import p_vis_keypoints, p_vis_3d_multiple_skeleton
from utils.RootUtils import r_process_bbox, r_pixel2cam
from utils.RootVis import r_vis_keypoints, r_vis_3d_skeleton


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 'n', print("Using CPU")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    return args

#loading models
args = parse_args()

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
if args.gpu_ids != 'n':
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

r_cfg.set_args(args.gpu_ids)
cudnn.benchmark = True

#root snapshot load
r_model_path = './snapshot_RootNet.pth.tar'
assert osp.exists(r_model_path), 'Cannot find model at ' + r_model_path
r_model = get_root_net(r_cfg, False)
if r_cfg.gpu_ids != 'n':
    r_model = DataParallel(r_model).cuda()
    ckpt = torch.load(r_model_path)
else:
    r_model = DataParallel(r_model)
    ckpt = torch.load(r_model_path,  map_location=torch.device('cpu'))
r_model.load_state_dict(ckpt['network'])
r_model.eval()

# MuCo joint set
joint_num = 21
joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20) )
skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )

p_cfg.set_args(args.gpu_ids)

#pose snapshot load
p_model_path = './snapshot_PoseNet.pth.tar'
assert osp.exists(p_model_path), 'Cannot find model at ' + p_model_path
p_model = get_pose_net(p_cfg, False, joint_num)
if p_cfg.gpu_ids != 'n':
    p_model = DataParallel(p_model).cuda()
    ckpt = torch.load(p_model_path)
else:
    p_model = DataParallel(p_model)
    ckpt = torch.load(p_model_path, map_location=torch.device('cpu'))
p_model.load_state_dict(ckpt['network'])
p_model.eval()

p_cfg.set_args(args.gpu_ids)

# Consolidate overlapping bounding boxes
def consolidate_bounding_boxes(boxes, threshold=0.5):
    def iou(box1, box2):
        # Calculate intersection over union (IoU) between two boxes
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    consolidated = []
    while boxes:
        box = boxes.pop(0)
        to_merge = [box]
        for other_box in boxes[:]:
            if iou(box, other_box) > threshold:
                to_merge.append(other_box)
                boxes.remove(other_box)
        # Merge boxes by calculating the bounding box of best fit
        x_coords = [b[0] for b in to_merge] + [b[0] + b[2] for b in to_merge]
        y_coords = [b[1] for b in to_merge] + [b[1] + b[3] for b in to_merge]
        x_min, y_min = min(x_coords), min(y_coords)
        x_max, y_max = max(x_coords), max(y_coords)
        consolidated.append((x_min, y_min, x_max - x_min, y_max - y_min))
    return consolidated

# Load YOLO model
def bbox_extractor(image):
# Get image dimensions
    (height, width) = image.shape[:2]

# Define the neural network input
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

# Perform forward propagation
    output_layer_name = net.getUnconnectedOutLayersNames()
    output_layers = net.forward(output_layer_name)

# Initialize list of detected people
    people = []

# Loop over the output layers
    for output in output_layers:
    # Loop over the detections
        for detection in output:
        # Extract the class ID and confidence of the current detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

        # Only keep detections with a high confidence
            if class_id == 0 and confidence > 0.5:
            # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

            # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

            # Add the detection to the list of people
                people.append((x, y, w, h))
    people = consolidate_bounding_boxes(people)
    return people

def extract_root_depth_from_image(original_img, vis = False):
# prepare input image
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=r_cfg.pixel_mean, std=r_cfg.pixel_std)])
    original_img_height, original_img_width = original_img.shape[:2]

# prepare bbox for each human
    bbox_list = bbox_extractor(original_img)
    person_num = len(bbox_list)
    #print(person_num)

# normalized camera intrinsics
    focal = [1500, 1500] # x-axis, y-axis
    princpt = [original_img_width/2, original_img_height/2] # x-axis, y-axis
    #print('focal length: (' + str(focal[0]) + ', ' + str(focal[1]) + ')')
    #print('principal points: (' + str(princpt[0]) + ', ' + str(princpt[1]) + ')')

    roots = []
# for cropped and resized human image, forward it to RootNet
    for n in range(person_num):
        bbox = r_process_bbox(np.array(bbox_list[n]), original_img_width, original_img_height)
        img, img2bb_trans = r_generate_patch_image(original_img, bbox, False, 0.0) 
        img = transform(img)[None,:,:,:]
        k_value = np.array([math.sqrt(r_cfg.bbox_real[0]*r_cfg.bbox_real[1]*focal[0]*focal[1]/(bbox[2]*bbox[3]))]).astype(np.float32)
        k_value = torch.FloatTensor([k_value])[None,:]

    # forward
        with torch.no_grad():
            root_3d = r_model(img, k_value) # x,y: pixel, z: root-relative depth (mm)
        img = img[0].cpu().numpy()
        root_3d = root_3d[0].cpu().numpy()

    # save output in 2D space (x,y: pixel)
        if vis:
            vis_img = img.copy()
            vis_img = vis_img * np.array(r_cfg.pixel_std).reshape(3,1,1) + np.array(r_cfg.pixel_mean).reshape(3,1,1)
            vis_img = vis_img.astype(np.uint8)
            vis_img = vis_img[::-1, :, :]
            vis_img = np.transpose(vis_img,(1,2,0)).copy()
            vis_root = np.zeros((2))
            vis_root[0] = root_3d[0] / r_cfg.output_shape[1] * r_cfg.input_shape[1]
            vis_root[1] = root_3d[1] / r_cfg.output_shape[0] * r_cfg.input_shape[0]
            cv2.circle(vis_img, (int(vis_root[0]), int(vis_root[1])), radius=5, color=(0,255,0), thickness=-1, lineType=cv2.LINE_AA)
            cv2.imwrite('output_root_2d_' + str(n) + '.jpg', vis_img)
    
        #print('Root joint depth: ' + str(root_3d[2]) + ' mm')
        roots.append(root_3d[2])
    return roots

def extract_pose(original_img, vis2D = False, vis3D = False):

# prepare input image
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=p_cfg.pixel_mean, std=p_cfg.pixel_std)])
    original_img_height, original_img_width = original_img.shape[:2]

# prepare bbox
    bbox_list = bbox_extractor(original_img)
    root_depth_list = extract_root_depth_from_image(original_img) # obtain this from RootNet (https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/tree/master/demo)
    assert len(bbox_list) == len(root_depth_list)
    person_num = len(bbox_list)

# normalized camera intrinsics
    focal = [1500, 1500] # x-axis, y-axis
    princpt = [original_img_width/2, original_img_height/2] # x-axis, y-axis
    #print('focal length: (' + str(focal[0]) + ', ' + str(focal[1]) + ')')
    #print('principal points: (' + str(princpt[0]) + ', ' + str(princpt[1]) + ')')

# for each cropped and resized human image, forward it to PoseNet
    output_pose_2d_list = []
    output_pose_3d_list = []
    for n in range(person_num):
        bbox = p_process_bbox(np.array(bbox_list[n]), original_img_width, original_img_height)
        img, img2bb_trans = p_generate_patch_image(original_img, bbox, False, 1.0, 0.0, False) 
        if p_cfg.gpu_ids != 'n':
            img = transform(img).cuda()[None,:,:,:]
        else:
            img = transform(img)[None,:,:,:]

    # forward
        with torch.no_grad():
            pose_3d = p_model(img) # x,y: pixel, z: root-relative depth (mm)

    # inverse affine transform (restore the crop and resize)
        pose_3d = pose_3d[0].cpu().numpy()
        pose_3d[:,0] = pose_3d[:,0] / p_cfg.output_shape[1] * p_cfg.input_shape[1]
        pose_3d[:,1] = pose_3d[:,1] / p_cfg.output_shape[0] * p_cfg.input_shape[0]
        pose_3d_xy1 = np.concatenate((pose_3d[:,:2], np.ones_like(pose_3d[:,:1])),1)
        img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0,0,1]).reshape(1,3)))
        pose_3d[:,:2] = np.dot(np.linalg.inv(img2bb_trans_001), pose_3d_xy1.transpose(1,0)).transpose(1,0)[:,:2]
        output_pose_2d_list.append(pose_3d[:,:2].copy())
    
    # root-relative discretized depth -> absolute continuous depth
        pose_3d[:,2] = (pose_3d[:,2] / p_cfg.depth_dim * 2 - 1) * (p_cfg.bbox_3d_shape[0]/2) + root_depth_list[n]
        pose_3d = p_pixel2cam(pose_3d, focal, princpt)
        output_pose_3d_list.append(pose_3d.copy())

# visualize 2d poses
    if vis2D:
        vis_img = original_img.copy()
        for n in range(person_num):
            vis_kps = np.zeros((3,joint_num))
            vis_kps[0,:] = output_pose_2d_list[n][:,0]
            vis_kps[1,:] = output_pose_2d_list[n][:,1]
            vis_kps[2,:] = 1
            vis_img = p_vis_keypoints(vis_img, vis_kps, skeleton)
        cv2.imwrite('output_pose_2d.jpg', vis_img)

    if vis3D:
        vis_kps = np.array(output_pose_3d_list)
        p_vis_3d_multiple_skeleton(vis_kps, np.ones_like(vis_kps), skeleton, 'output_pose_3d (x,y,z: camera-centered. mm.)')
    
    return output_pose_2d_list, output_pose_3d_list

def process_video(input_video_path, output_video_path_2d, output_video_path_3d):
    skelDim = cv2.imread('3d_multiple_skeleton.png')
    out_3d_height, out_3d_width, _ = skelDim.shape
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create VideoWriter objects for the output videos
    out_2d = cv2.VideoWriter(output_video_path_2d, fourcc, fps, (frame_width, frame_height))
    out_3d = cv2.VideoWriter(output_video_path_3d, fourcc, fps, (out_3d_width, out_3d_height))

    # Start timing
    start_time = time.time()
    frame_times = []

    i = 1
    while True:
        frame_start_time = time.time()
        print(f"Processing frame {i}/{length}")
        ret, frame = cap.read()
        if not ret:
            break

        # apply pose extraction and visualization
        _, _ = extract_pose(frame, vis2D=True, vis3D=True)

        # load the frames with the 2D and 3D pose visualizations applied
        vis_frame_2d = cv2.imread('output_pose_2d.jpg')
        vis_frame_3d = cv2.imread('3d_multiple_skeleton.png')

        # write the processed frames to the respective output videos
        if vis_frame_2d is not None:
            out_2d.write(vis_frame_2d)
        if vis_frame_3d is not None:
            out_3d.write(vis_frame_3d)

        # record frame processing time
        frame_end_time = time.time()
        frame_times.append(frame_end_time - frame_start_time)
        i += 1

    # end time
    total_time = time.time() - start_time
    avg_time_per_frame = sum(frame_times) / len(frame_times) if frame_times else 0
            

    # Release videos
    cap.release()
    out_2d.release()
    out_3d.release()
    print(f"Processed 2D video saved to {output_video_path_2d}")
    print(f"Processed 3D video saved to {output_video_path_3d}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per frame: {avg_time_per_frame:.2f} seconds")

process_video('input_video.mp4', 'output_video_2d.mp4', 'output_video_3d.mp4')