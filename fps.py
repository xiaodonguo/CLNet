# TODO - 计算模型的推理时间
def calcTime():

    import numpy as np
    from torchvision.models import resnet50
    import torch
    from torch.backends import cudnn
    import tqdm
    from Teacher_model import Teacher_model
    # from model_others.MMSMCNet import nation
    '''  导入你的模型
    from module.amsnet import amsnet, anet, msnet, iresnet18, anet2, iresnet2, amsnet2
    from module.resnet import resnet18, resnet34
    from module.alexnet import AlexNet
    from module.vgg import vgg
    from module.lenet import LeNet
    from module.googLenet import GoogLeNet
    from module.ivgg import iVGG
    '''


    cudnn.benchmark = True

    device = 'cuda:0'

    repetitions = 1000
    # model = nation().to(device)
    # model = Teacher().to(device)
    model = Teacher_model(9).to(device)

    dummy_input = torch.rand(2, 4, 480, 640).to(device)
    # dummy_input1 = torch.rand(2, 3, 480, 640).to(device)
    # dummy_input2 = torch.rand(2, teacher, 256, 256).to(device)
    # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)

    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    torch.cuda.synchronize()

    # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # 初始化一个时间容器
    timings = np.zeros((repetitions, 1))

    print('testing ...\n')
    with torch.no_grad():
        for rep in tqdm.tqdm(range(repetitions)):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()  # 等待GPU任务完成
            curr_time = starter.elapsed_time(ender)  # 从 starter 到 ender 之间用时,单位为毫秒
            timings[rep] = curr_time

    avg = timings.sum() / repetitions
    print('\navg={}\n'.format(avg))

if __name__ == '__main__':
    calcTime()