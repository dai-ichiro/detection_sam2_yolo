import os
import glob
import cv2
import torch
import numpy as np
from argparse import ArgumentParser

from sam2.build_sam import build_sam2_video_predictor
from ultralytics import YOLO

parser = ArgumentParser()
parser.add_argument('--videos_dir', type=str, default='videos', help='video folder name')
parser.add_argument('--epochs', type=int, default=4, help='total training epochs')
parser.add_argument('--batch', type=int, default=8, help='total batch size')
parser.add_argument('--weights', type=str, default='yolo26m.pt', help='initial weights path')
parser.add_argument('--imgsz', type=int, default=640, help='image size for training')
parser.add_argument('--sam2_checkpoint', type=str, default='models/sam2.1_hiera_large.pt', help='SAM2 checkpoint path')
parser.add_argument('--sam2_config_dir', type=str, default='models', help='SAM2 hydra config directory')
parser.add_argument('--sam2_model_cfg', type=str, default='sam2.1_hiera_l', help='SAM2 model config name')
parser.add_argument('--mode', type=str, default='both', choices=['tracking', 'yolo_train', 'both'], help='execution mode')
args = parser.parse_args()


def tracking():
    class_list = glob.glob(os.path.join(args.videos_dir, '*'))
    class_num = len(class_list)
    print(f'class count = {class_num}')

    video_list = []
    classname_list = []

    for each_class in class_list:
        if os.path.isdir(each_class):
            classname_list.append(os.path.basename(each_class))
            video_list.append(glob.glob(os.path.join(each_class, '*')))
        else:
            classname_list.append(os.path.splitext(os.path.basename(each_class))[0])
            video_list.append([each_class])

    for i, classname in enumerate(classname_list):
        print(f'class {i}: {classname}')
    for i, videos in enumerate(video_list):
        print(f'videos of class {i}: {", ".join(videos)}')

    out_path = 'train_data'
    train_images_dir = os.path.join(out_path, 'images', 'train')
    train_labels_dir = os.path.join(out_path, 'labels', 'train')
    tracked_videos_dir = os.path.join(out_path, 'tracked_videos')
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(tracked_videos_dir, exist_ok=True)

    # Collect initial ROIs from user for all videos before tracking starts
    init_rect_list = []
    for videos_in_each_class in video_list:
        rects_for_class = []
        for video in videos_in_each_class:
            cap = cv2.VideoCapture(video)
            _, img = cap.read()
            cap.release()
            source_window = "draw_rectangle"
            cv2.namedWindow(source_window)
            rect = cv2.selectROI(source_window, img, False, False)
            # SAM2 requires [xmin, ymin, xmax, ymax]
            initial_box = np.array(
                [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]],
                dtype=np.float32
            )
            rects_for_class.append(initial_box)
            cv2.destroyAllWindows()
        init_rect_list.append(rects_for_class)

    # Initialize SAM2 predictor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    from hydra import initialize_config_dir
    import hydra
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=os.path.abspath(args.sam2_config_dir), version_base=None)

    predictor = build_sam2_video_predictor(args.sam2_model_cfg, args.sam2_checkpoint, device=device)

    print('start making dataset...')

    for class_index, videos in enumerate(video_list):
        for video_index, video in enumerate(videos):
            # Read all frames via OpenCV to save images later
            cap = cv2.VideoCapture(video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            if not frames:
                print(f'Warning: no frames read from {video}, skipping.')
                continue

            h, w = frames[0].shape[:2]

            # Prepare VideoWriter for bbox-annotated output
            video_basename = os.path.splitext(os.path.basename(video))[0]
            out_video_path = os.path.join(
                tracked_videos_dir,
                '%d_%d_%s_tracked.mp4' % (class_index, video_index, video_basename)
            )
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

            # Init SAM2 state and register the initial bounding box
            inference_state = predictor.init_state(video_path=video)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                box=init_rect_list[class_index][video_index],
            )

            # Propagate tracking across the video
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                if out_frame_idx >= len(frames):
                    continue

                # Extract bbox from the predicted mask
                bbox = None
                for i in range(len(out_obj_ids)):
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy().astype(np.uint8)
                    if mask.ndim == 3:
                        mask = mask[0]
                    if np.any(mask):
                        y_indices, x_indices = np.where(mask)
                        xmin = int(np.min(x_indices))
                        xmax = int(np.max(x_indices))
                        ymin = int(np.min(y_indices))
                        ymax = int(np.max(y_indices))
                        bbox = (xmin, ymin, xmax, ymax)
                        break  # one object per video

                frame = frames[out_frame_idx]

                if bbox is None:
                    print(f'Frame {out_frame_idx:04d}: object not detected, skipping.')
                    video_writer.write(frame)
                    continue

                center_x = ((bbox[0] + bbox[2]) / 2) / w
                center_y = ((bbox[1] + bbox[3]) / 2) / h
                bbox_w = (bbox[2] - bbox[0]) / w
                bbox_h = (bbox[3] - bbox[1]) / h

                filename = '%d_%d_%06d' % (class_index, video_index, out_frame_idx)

                cv2.imwrite(os.path.join(train_images_dir, filename + '.jpg'), frame)

                with open(os.path.join(train_labels_dir, filename + '.txt'), 'w') as f:
                    f.write('%d %f %f %f %f' % (class_index, center_x, center_y, bbox_w, bbox_h))

                # Draw bbox on a copy and write to output video
                annotated = frame.copy()
                cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(annotated, classname_list[class_index],
                            (bbox[0], bbox[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
                video_writer.write(annotated)

            video_writer.release()
            print(f'Saved tracked video: {out_video_path}')
            predictor.reset_state(inference_state)

    abs_out_path = os.path.abspath(out_path)
    with open('train.yaml', 'w', encoding='cp932') as f:
        f.write(f'train: {abs_out_path}/images/train\n')
        f.write(f'val: {abs_out_path}/images/train\n')
        f.write('nc: %d\n' % class_num)
        f.write('names: [')
        f.write(', '.join(['\'' + x + '\'' for x in classname_list]))
        f.write(']')


def yolo_train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(args.weights)
    model.train(
        data='train.yaml',
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
    )


if __name__ == '__main__':
    if args.mode in ('tracking', 'both'):
        tracking()
        print('finish making dataset.')
    if args.mode in ('yolo_train', 'both'):
        print('start training...')
        yolo_train()
        print('finish training.')
