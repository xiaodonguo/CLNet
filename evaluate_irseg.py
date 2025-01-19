import os
import time
from tqdm import tqdm
from PIL import Image
import json

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from toolbox import get_dataset
from toolbox import get_model
from toolbox.metrics import averageMeter, runningScore
from toolbox import class_to_RGB, load_ckpt, save_ckpt
from toolbox.datasets.irseg import IRSeg
# from proposed.teacher.b4.wo_DGSD import Teacher
# from proposed.teacher.b4.model5 import Teacher


def evaluate(logdir, save_predict=False, options=['val', 'test', 'test_day', 'test_night'], prefix=''):
    # 加载配置文件cfg
    cfg = None
    for file in os.listdir(logdir):
        if file.endswith('.json'):
            with open(os.path.join(logdir, file), 'r') as fp:
                cfg = json.load(fp)
    assert cfg is not None

    device = torch.device('cuda:2')

    loaders = []
    for opt in options:
        dataset = IRSeg(cfg, mode=opt)
        loaders.append((opt, DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'])))
        cmap = dataset.cmap

    # model = get_model(cfg).to(device)
    # model = Teacher().to(device)
    model = Student().to(device)
    model.load_state_dict(torch.load('/home/guoxiaodong/code/seg/Semantic_Segmentation_Street_Scenes/run/ablation2/DSRD/193_model.pth'))
    running_metrics_val = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    time_meter = averageMeter()
    save_path = os.path.join('result', 'ablation2/DSRD/')
    if not os.path.exists(save_path) and save_predict:
        os.makedirs(save_path)

    for name, test_loader in loaders:
        running_metrics_val.reset()
        print('#'*50 + '    ' + name+prefix + '    ' + '#'*50)
        with torch.no_grad():
            model.eval()
            for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
                time_start = time.time()
                if cfg['inputs'] == 'rgb':
                    image = sample['image'].to(device)
                    label = sample['label'].to(device)
                    predict = model(image)
                else:
                    image = sample['image'].to(device)
                    depth = sample['depth'].to(device)
                    label = sample['label'].to(device)
                    predict = model(image, depth)[6]

                predict = predict.max(1)[1].cpu().numpy()  # [1, h, w] 按照第一个维度求最大值，并返回最大值对应的索引
                label = label.cpu().numpy()
                running_metrics_val.update(label, predict)

                time_meter.update(time.time() - time_start, n=image.size(0))

                if save_predict:
                    predict = predict.squeeze(0)  # [1, h, w] -> [h, w]
                    predict = class_to_RGB(predict, N=len(cmap), cmap=cmap)  # 如果数据集没有给定cmap,使用默认cmap
                    predict = Image.fromarray(predict)
                    predict.save(os.path.join(save_path, sample['label_path'][0]))

        metrics = running_metrics_val.get_scores()
        print('overall metrics .....')
        for k, v in metrics[0].items():
            print(k, f'{v:.3f}')

        print('iou for each class .....')
        for k, v in metrics[1].items():
            print(k, f'{v:.3f}')
        print('acc for each class .....')
        for k, v in metrics[2].items():
            print(k, f'{v:.3f}')

        print('inference time per image: ', time_meter.avg)
        print('inference fps: ', 1 / time_meter.avg)
        print(f'{metrics[0]["class_acc: "]:.3f}', f'{metrics[0]["mIou: "]:.3f}')



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="evaluate")
    parser.add_argument("--logdir", type=str, default="/home/guoxiaodong/code/seg/Semantic_Segmentation_Street_Scenes/run/teacher_student_KD/teacher")
    parser.add_argument("-s", type=bool, default=True, help="save predict or not")
    args = parser.parse_args()
    evaluate(args.logdir, save_predict=args.s, options=['val'], prefix='')
