import torch
from Semantic_Segmentation_Street_Scenes.backbone.mix_transformer import mit_b2
import torch.nn as nn
from Semantic_Segmentation_Street_Scenes.toolbox import setup_seed
setup_seed(33)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.rgb = mit_b2()
        # self.t = mit_b2()
        if self.training:
            self.rgb.load_state_dict(torch.load(
                "/home/guoxiaodong/code/seg/Semantic_Segmentation_Street_Scenes/backbone/SegFormer_master/weight/mit_b2.pth"),
                                     strict=False)
            # self.t.load_state_dict(torch.load(
            #     "/home/guoxiaodong/code/seg/Semantic_Segmentation_Street_Scenes/backbone/SegFormer_master/weight/mit_b2.pth"),
            #                        strict=False)
        self.fusion1 = Fusion(64, 120, 160)
        self.fusion2 = Fusion(128, 60, 80)
        self.fusion3 = Fusion(320, 30, 40)
        self.fusion4 = Fusion(512, 15, 20)
        self.bound1 = Boundary()
        self.bound2 = Boundary()
        self.bound3 = Boundary()
        self.binary1 = Binary()
        self.binary2 = Binary()
        self.binary3 = Binary()
        self.semantic1 = Semantic()
        self.semantic2 = Semantic()
        self.semantic3 = Semantic()
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.cbl_out = nn.Sequential(
                                   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=64, out_channels=9, kernel_size=3, stride=1, padding=1),
                                   )
        self.cbl_b1 = BasicConv2d(64, 2, 3, 1, 1)
        self.cbl_b2 = BasicConv2d(64, 2, 3, 1, 1)
        self.cbl_b3 = BasicConv2d(64, 2, 3, 1, 1)

        self.cbl_S1 = BasicConv2d(64, 2, 3, 1, 1)
        self.cbl_S2 = BasicConv2d(64, 2, 3, 1, 1)
        self.cbl_S3 = BasicConv2d(64, 2, 3, 1, 1)

    def forward(self, rgb, t):
        # t = rgb
        rgb1, rgb2, rgb3, rgb4 = self.rgb(rgb)
        # rgb1, t2, t3, t4 = self.t(t)
        t1, t2, t3, t4 = self.rgb(t)
        fusion1 = self.fusion1(rgb1, t1, former=None, NO=1)
        fusion2 = self.fusion2(rgb2, t2, fusion1, NO=2)
        fusion3 = self.fusion3(rgb3, t3, fusion2, NO=3)
        fusion4 = self.fusion4(rgb4, t4, fusion3, NO=4)
        fusion = [fusion1, fusion2, fusion3, fusion4]
        bound1 = self.bound1(fusion[0], fusion[1], fusion[3], NO=1)
        binary1 = self.binary1(fusion[2], fusion[3], fusion[3], NO=1)
        semantic1 = self.semantic1(bound1, binary1, fusion[3], NO=1)
        bound1_out = self.cbl_b1(self.up8(bound1))
        binary1_out = self.cbl_S1(self.up8(binary1))

        bound2 = self.bound2(fusion[0], fusion[1], semantic1, NO=2)
        binary2 = self.binary1(fusion[2], fusion[3], semantic1, NO=2)
        semantic2 = self.semantic1(bound2, binary2, semantic1, NO=2)
        bound2_out = self.cbl_b2(self.up4(bound2))
        binary2_out = self.cbl_S2(self.up4(binary2))

        bound3 = self.bound3(fusion[0], fusion[1], semantic2, NO=3)
        binary3 = self.binary1(fusion[2], fusion[3], semantic2, NO=3)
        semantic3 = self.semantic1(bound3, binary3, semantic2, NO=3)
        bound3_out = self.cbl_b3(self.up2(bound3))
        binary3_out = self.cbl_S3(self.up2(binary3))
        semantic_out = self.cbl_out(self.up2(semantic3))

        return bound1_out, bound2_out, bound3_out, binary1_out, binary2_out, binary3_out, semantic_out

class Fusion(nn.Module):
    def __init__(self, inplanes, H, W):
        super(Fusion, self).__init__()
        self.SA = SAM()
        self.CA = CAM(2 * inplanes, 2 * inplanes, 16)
        self.conv1 = BasicConv2d(inplanes, 64, 3, 1, 1)
        self.conv2 = BasicConv2d(inplanes, 64, 3, 1, 1)
        self.dilation1 = Dilation()
        self.dilation2 = Dilation()
        self.interaction1 = Interaction()
        self.interaction2 = Interaction()
        self.interaction3 = Interaction()
        self.interaction4 = Interaction()
        self.conv3 = BasicConv2d(256, 64, 3, 1, 1)
        self.conv4_1 = BasicConv2d(64, 64, 3, 2, 1)
        self.conv4_2 = BasicConv2d(64, 64, 3, 1, 1)
        self.conv5_1 = BasicConv2d(64, 64, 3, 1, 1)
        self.conv5_2 = BasicConv2d(128, 64, 3, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(H // 2, W // 2))
    def forward(self, rgb, t, former, NO):
        if NO == 1 or NO == 2:
            rgb, t = self.SA(rgb, t)
        else:
            rgb, t = self.CA(rgb, t)
        p1 = rgb + t
        p2 = torch.mul(rgb, t)
        p1 = self.conv1(p1)
        p2 = self.conv2(p2)
        add1, add2, add3, add4 = self.dilation1(p1)
        mul1, mul2, mul3, mul4 = self.dilation2(p2)
        interaction1 = self.interaction1(add1, mul1, former=None, NO=1)
        interaction2 = self.interaction2(add2, mul2, former=interaction1, NO=2)
        interaction3 = self.interaction3(add3, mul3, former=interaction2, NO=3)
        interaction4 = self.interaction4(add4, mul4, former=interaction3, NO=4)
        dilation_out = torch.cat((interaction1, interaction2, interaction3, interaction4), dim=1)
        dilation_out = self.conv3(dilation_out)
        if NO == 4:
            out = p1 + dilation_out + p2
        else:
            out = self.pool(p1 + dilation_out + p2)
        if NO == 1:
            out = self.conv5_1(out)
        elif NO == 4:
            former = self.conv4_2(former)
            out = self.conv5_2(torch.cat((former, out), dim=1))
        else:
            former = self.conv4_1(former)
            out = self.conv5_2(torch.cat((former, out), dim=1))
        return out

class Interaction(nn.Module):
    def __init__(self):
        super(Interaction, self).__init__()
        self.conv1 = BasicConv2d(64, 64, 3, 1, 1)
        self.conv2 = BasicConv2d(64, 64, 3, 1, 1)
        self.conv3_1 = BasicConv2d(128, 64, 3, 1, 1)
        self.conv3_2 = BasicConv2d(192, 64, 3, 1, 1)
    def forward(self, add, mul, former, NO):
        add = self.conv1(add)
        mul = self.conv2(mul)
        if NO == 1:
            out = torch.cat((add, mul), dim=1)
            out = self.conv3_1(out)
        else:
            out = torch.cat((add, mul, former), dim=1)
            out = self.conv3_2(out)
        return out

class Dilation(nn.Module):
    def __init__(self):
        super(Dilation, self).__init__()
        self.dilation1 = DSC(64, 64, 3, 3, 3)
        self.dilation2 = DSC(64, 64, 3, 5, 5)
        self.dilation3 = DSC(64, 64, 3, 7, 7)
        self.dilation4 = DSC(64, 64, 3, 9, 9)
    def forward(self, input):
        out1 = self.dilation1(input)
        out2 = self.dilation2(input)
        out3 = self.dilation3(input)
        out4 = self.dilation4(input)
        return out1, out2, out3, out4

class Boundary(nn.Module):
    def __init__(self):
        super(Boundary, self).__init__()
        self.conv1 = BasicConv2d(64, 64, 3, 1, 1)
        self.conv2 = BasicConv2d(64, 64, 3, 1, 1)
        self.conv3 = BasicConv2d(64, 64, 3, 1, 1)
        self.conv4 = BasicConv2d(64, 64, 3, 1, 1)
        self.conv5 = BasicConv2d(128, 64, 3, 1, 1)
        self.conv6 = BasicConv2d(64, 64, 3, 1, 1)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, e1, e2, di, NO):
        if NO == 1:
            e1 = self.conv1(e1)
            e2 = self.conv2(self.up2(e2))  # 60 X 80
            di = self.conv3(self.up4(di))
        if NO == 2:
            e1 = self.conv1(self.up2(e1))  # 120 X 160
            e2 = self.conv2(self.up4(e2))
            di = self.conv3(self.up2(di))
        if NO == 3:
            e1 = self.conv1(self.up4(e1))  # 240 X 320
            e2 = self.conv2(self.up8(e2))
            di = self.conv3(self.up2(di))
        e12 = torch.mul(e1, e2)
        e12 = self.conv4(e12)
        di = self.conv5(torch.cat((di, e1), dim=1))
        out = self.conv6(di + e12)
        return out

class Binary(nn.Module):
    def __init__(self):
        super(Binary, self).__init__()
        self.conv1 = BasicConv2d(64, 64, 3, 1, 1)
        self.conv2 = BasicConv2d(64, 64, 3, 1, 1)
        self.conv3 = BasicConv2d(128, 64, 3, 1, 1)
        self.conv4 = BasicConv2d(128, 64, 3, 1, 1)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
    def forward(self, e3, e4, di, NO):
        if NO == 1:
            e3 = self.conv1(self.up4(e3))  # 60, 80
            e4 = self.conv2(self.up4(e4))
            di = self.up4(di)
        if NO == 2:
            e3 = self.conv1(self.up8(e3))  # 120, 160
            e4 = self.conv2(self.up8(e4))
            di = self.up2(di)
        if NO == 3:
            e3 = self.conv1(self.up16(e3))  # 240, 320
            e4 = self.conv2(self.up16(e4))
            di = self.up2(di)
        e34 = torch.mul(e3+e4, self.conv3(torch.cat((e3, e4), dim=1)))
        out = self.conv4(torch.cat((e34, di), dim=1))
        return out

class Semantic(nn.Module):
    def __init__(self):
        super(Semantic, self).__init__()
        self.conv1 = BasicConv2d(64, 64, 3, 1, 1)
        self.conv2 = BasicConv2d(64, 64, 3, 1, 1)
        self.conv3 = BasicConv2d(64, 64, 3, 1, 1)
        self.conv4 = BasicConv2d(192, 64, 3, 1, 1)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, bound, binary, di, NO):
        if NO == 1:
            di = self.conv1(self.up4(di))
        else:
            di = self.conv1(self.up2(di))
        binary = self.conv2(binary)
        bound = self.conv3(bound)
        out = self.conv4(torch.cat((binary, bound, di), dim=1))

        return out

class BasicConv2d(nn.Module):
    def __init__(self, inplanes, outplanes, kernelsize, stride=1, padding=0, dialation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernelsize, padding=padding, stride=stride,
                              dilation=dialation, bias=False)
        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DSC(nn.Module):
    def __init__(self, inchannels, outchannels, kenelsize, padding, dilation):
        super(DSC, self).__init__()
        self.depthwiseConv = nn.Conv2d(inchannels, inchannels, kenelsize, groups=inchannels, padding=padding, dilation=dilation)
        self.pointwiseConv = nn.Conv2d(inchannels, outchannels, 1)
        self.BN = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.depthwiseConv(x)
        x = self.pointwiseConv(x)
        x = self.relu(self.BN(x))
        return x

class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 7, 1, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb, t):
        max_out, _ = torch.max(torch.cat((rgb, t), dim=1), dim=1, keepdim=True)
        out1 = self.conv1(max_out)
        weight_rgb = self.sigmoid(out1)
        weight_t = 1 - weight_rgb
        rgb_out = rgb * weight_rgb
        t_out = t * weight_t
        return rgb_out, t_out

class CAM(nn.Module):
    def __init__(self, inplanes, outplanes, ratio):
        super(CAM, self).__init__()
        self.inplanes = inplanes // 2
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.FC1 = nn.Conv2d(inplanes, inplanes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.FC2 = nn.Conv2d(inplanes // ratio, outplanes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, rgb, t):
        out = self.FC2(self.relu1(self.FC1(self.maxpool(torch.cat((rgb, t), dim=1)))))
        channel_weight = self.sigmoid(out)
        out1 = torch.mul(torch.cat((rgb, t), dim=1), channel_weight)
        rgb = out1[:, 0:self.inplanes, :, :]
        t = out1[:, self.inplanes:2*self.inplanes, :, :]
        return rgb, t

if __name__ == '__main__':

    rgb_input = torch.randn(2, 3, 480, 640)
    t_input = torch.randn(2, 3, 480, 640)
    model = Model()
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (3, 480, 640), as_strings=True, print_per_layer_stat=False)
    print('Flops ' + flops)
    print('Params ' + params)