import os
from tqdm import tqdm
import json

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from toolbox import get_model
import numpy as np
import cv2
from toolbox.datasets.irseg import IRSeg
from proposed.teacher.b4.visualize import Teacher


def evaluate(logdir, save_predict=False, options=['train', 'val', 'test', 'test_day', 'test_night'], prefix=''):
    # 加载配置文件cfg
    cfg = None
    for file in os.listdir(logdir):
        if file.endswith('.json'):
            with open(os.path.join(logdir, file), 'r') as fp:
                cfg = json.load(fp)
    assert cfg is not None
    device = torch.device("cuda:6")
    loaders = []
    for opt in options:
        dataset = IRSeg(cfg, mode=opt)
        loaders.append((opt, DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'])))
    model = Teacher().to(device)
    model.load_state_dict(torch.load('/home/guoxiaodong/code/seg/Semantic_Segmentation_Street_Scenes/run/teacher_student_KD/teacher/163model.pth'))
    save_path = 'feature_maps/cat/'
    if not os.path.exists(save_path) and save_predict:
        os.makedirs(save_path, exist_ok=True)

    for name, test_loader in loaders:
        with torch.no_grad():
            model.eval()
            for i, sample in enumerate(test_loader):
                if cfg['inputs'] == 'rgb':
                    image = sample['image'].to(device)
                    label = sample['label'].to(device)
                    predict = model(image)[0]
                else:
                    image = sample['image'].to(device)
                    depth = sample['depth'].to(device)
                    label = sample['label'].to(device)
                    predict = model(image, depth)[4]
                out = F.interpolate(predict, size=(480, 640), mode="bilinear", align_corners=False)
                out_img = out.cpu().detach().numpy()
                out_img = np.max(out_img, axis=1).reshape(480, 640)
                out_img = (((out_img - np.min(out_img)) / (np.max(out_img) - np.min(out_img)))*255).astype(np.uint8)
                out_img = cv2.applyColorMap(out_img, cv2.COLORMAP_JET)
                cv2.imwrite(save_path + '/' + sample['label_path'][0], out_img)

                print(save_path + sample['label_path'][0])
                # if i == 10:
                #     break






if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="evaluate")
    parser.add_argument("--logdir", type=str, default="/home/guoxiaodong/code/seg/Semantic_Segmentation_Street_Scenes/run/teacher_student_KD/teacher")
    parser.add_argument("-s", type=bool, default=True, help="save predict or not")

    args = parser.parse_args()
    evaluate(args.logdir, save_predict=args.s, options=['val'], prefix='')



