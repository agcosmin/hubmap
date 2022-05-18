import torch
import torch.nn
import torch.fft


class ResidualModule(torch.nn.Module):
    def __init__(self, num_channels):
        super(ResidualModule, self).__init__()

        self.conv0 = torch.nn.Conv2d(in_channels=num_channels,
                                     out_channels=num_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     bias=True)

        self.conv1 = torch.nn.Conv2d(in_channels=num_channels,
                                     out_channels=num_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     bias=True)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv0_relu = torch.nn.functional.relu(conv0, inplace=True)
        conv1 = self.conv1(conv0_relu)
        residual = x + conv1
        residual_relu = torch.nn.functional.relu(residual, inplace=True)

        return residual_relu


class YOLOv3Module(torch.nn.Module):
    def __init__(self, num_channels):
        super(YOLOv3Module, self).__init__()

        self.conv0 = torch.nn.Conv2d(in_channels=num_channels,
                                     out_channels=num_channels // 2,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     bias=True)

        self.conv1 = torch.nn.Conv2d(in_channels=num_channels // 2,
                                     out_channels=num_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     bias=True)

        self.residual = ResidualModule(num_channels)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv0_relu = torch.nn.functional.relu(conv0, inplace=True)
        conv1 = self.conv1(conv0_relu)
        conv1_relu = torch.nn.functional.relu(conv1, inplace=True)
        residual = self.residual(conv1_relu)

        return residual


class GlomerusDetector(torch.nn.Module):
    def __init__(self):
        super(GlomerusDetector, self).__init__()

        self.resizer = torch.nn.Conv2d(in_channels=3,
                                       out_channels=1,
                                       kernel_size=9,
                                       stride=4,
                                       padding=4,
                                       bias=False)

        self.conv0 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=32,
                                     kernel_size=3,
                                     padding=1,
                                     stride=1,
                                     bias=True)

        self.conv1 = torch.nn.Conv2d(in_channels=32,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     bias=True)

        self.yolo2 = YOLOv3Module(64)

        self.conv3 = torch.nn.Conv2d(in_channels=64,
                                     out_channels=128,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     bias=True)

        self.yolo4 = YOLOv3Module(128)
        self.yolo5 = YOLOv3Module(128)

        self.conv6 = torch.nn.Conv2d(in_channels=128,
                                     out_channels=256,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     bias=True)

        self.yolo7 = YOLOv3Module(256)
        self.yolo8 = YOLOv3Mdule(256)
        self.yolo9 = YOLOv3Module(256)
        self.yolo10 = YOLOv3Module(256)
        self.yolo11 = YOLOv3Module(256)
        self.yolo12 = YOLOv3Module(256)
        self.yolo13 = YOLOv3Module(256)
        self.yolo14 = YOLOv3Module(256)

        self.conv15 = torch.nn.Conv2d(in_channels=256,
                                      out_channels=512,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      bias=True)

        self.yolo16 = YOLOv3Module(512)
        self.yolo17 = YOLOv3Module(512)
        self.yolo18 = YOLOv3Module(512)
        self.yolo19 = YOLOv3Module(512)
        self.yolo20 = YOLOv3Module(512)
        self.yolo21 = YOLOv3Module(512)
        self.yolo22 = YOLOv3Module(512)
        self.yolo23 = YOLOv3Module(512)

        self.conv24 = torch.nn.Conv2d(in_channels=512,
                                      out_channels=1024,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      bias=True)

        self.yolo25 = YOLOv3Module(1024)
        self.yolo26 = YOLOv3Module(1024)
        self.yolo27 = YOLOv3Module(1024)
        self.yolo28 = YOLOv3Module(1024)

        self.tconv29 = torch.nn.ConvTranspose2d(in_channels=1024,
                                                out_channels=512,
                                                kernel_size=2,
                                                stride=2,
                                                padding=0,
                                                bias=True)

        self.tconv30 = torch.nn.ConvTranspose2d(in_channels=512,
                                                out_channels=256,
                                                kernel_size=2,
                                                stride=2,
                                                padding=0,
                                                bias=True)

        self.tconv31 = torch.nn.ConvTranspose2d(in_channels=256,
                                                out_channels=128,
                                                kernel_size=2,
                                                stride=2,
                                                padding=0,
                                                bias=True)

        self.tconv32 = torch.nn.ConvTranspose2d(in_channels=128,
                                                out_channels=64,
                                                kernel_size=2,
                                                stride=2,
                                                padding=0,
                                                bias=True)

        self.tconv33 = torch.nn.ConvTranspose2d(in_channels=64,
                                                out_channels=64,
                                                kernel_size=2,
                                                stride=2,
                                                padding=0,
                                                bias=True)

        self.scores = torch.nn.ConvTranspose2d(in_channels=64,
                                               out_channels=1,
                                               kernel_size=4,
                                               stride=4,
                                               padding=0,
                                               bias=True)


    def forward(self, x):
        resizer = self.resizer(x)

        conv0 = self.conv0(resizer)
        conv0_relu = torch.nn.functional.relu(conv0, inplace=True)

        conv1 = self.conv1(conv0_relu)
        conv1_relu = torch.nn.functional.relu(conv1, inplace=True)

        yolo2 = self.yolo2(conv1_relu)

        conv3 = self.conv3(yolo2)
        conv3_relu = torch.nn.functional.relu(conv3, inplace=True)

        yolo4 = self.yolo4(conv3_relu)
        yolo5 = self.yolo5(yolo4)

        conv6 = self.conv6(yolo5)
        conv6_relu = torch.nn.functional.relu(conv6, inplace=True)

        yolo7 = self.yolo7(conv6_relu)
        yolo8 = self.yolo8(yolo7)
        yolo9 = self.yolo9(yolo8)
        yolo10 = self.yolo10(yolo9)
        yolo11 = self.yolo11(yolo10)
        yolo12 = self.yolo12(yolo11)
        yolo13 = self.yolo13(yolo12)
        yolo14 = self.yolo14(yolo13)

        conv15 = self.conv15(yolo14)
        conv15_relu = torch.nn.functional.relu(conv15, inplace=True)

        yolo16 = self.yolo16(conv15_relu)
        yolo17 = self.yolo17(yolo16)
        yolo18 = self.yolo18(yolo17)
        yolo19 = self.yolo19(yolo18)
        yolo20 = self.yolo20(yolo19)
        yolo21 = self.yolo21(yolo20)
        yolo22 = self.yolo22(yolo21)
        yolo23 = self.yolo23(yolo22)

        conv24 = self.conv24(yolo23)
        conv24_relu = torch.nn.functional.relu(conv24, inplace=True)

        yolo25 = self.yolo25(conv24_relu)
        yolo26 = self.yolo26(yolo25)
        yolo27 = self.yolo27(yolo26)
        yolo28 = self.yolo28(yolo27)

        tconv29 = self.tconv29(yolo28)
        tconv29_relu = torch.nn.functional.relu(tconv29, inplace=True)
        tconv29_residual = yolo23 - tconv29_relu
        tconv29_residual_relu = torch.nn.functional.relu(tconv29_residual,
                                                         inplace=True)

        tconv30 = self.tconv30(tconv29_residual_relu)
        tconv30_relu = torch.nn.functional.relu(tconv30, inplace=True)
        tconv30_residual = yolo14 - tconv30_relu
        tconv30_residual_relu = torch.nn.functional.relu(tconv30_residual,
                                                         inplace=True)

        tconv31 = self.tconv31(tconv30_residual_relu)
        tconv31_relu = torch.nn.functional.relu(tconv31, inplace=True)
        tconv31_residual = yolo5 - tconv31_relu
        tconv31_residual_relu = torch.nn.functional.relu(tconv31_residual,
                                                         inplace=True)

        tconv32 = self.tconv32(tconv31_residual_relu)
        tconv32_relu = torch.nn.functional.relu(tconv32, inplace=True)
        tconv32_residual = yolo2 - tconv32_relu
        tconv32_residual_relu = torch.nn.functional.relu(tconv32_residual,
                                                         inplace=True)

        tconv33 = self.tconv33(tconv32_residual_relu)
        tconv33_relu = torch.nn.functional.relu(tconv33, inplace=True)

        scores = self.scores(tconv33_relu)
        scores = torch.sigmoid(scores)

        return (scores, )


class GlomerusDetector6(torch.nn.Module):
    def __init__(self):
        super(GlomerusDetector6, self).__init__()

        self.resizer = torch.nn.Conv2d(in_channels=3,
                                       out_channels=1,
                                       kernel_size=9,
                                       stride=4,
                                       padding=4,
                                       bias=False)

        self.conv0 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=32,
                                     kernel_size=3,
                                     padding=1,
                                     stride=1,
                                     bias=True)

        self.conv1 = torch.nn.Conv2d(in_channels=32,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     bias=True)

        self.yolo2 = YOLOv3Module(64)

        self.conv3 = torch.nn.Conv2d(in_channels=64,
                                     out_channels=128,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     bias=True)

        self.yolo4 = YOLOv3Module(128)
        self.yolo5 = YOLOv3Module(128)

        self.conv6 = torch.nn.Conv2d(in_channels=128,
                                     out_channels=256,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     bias=True)

        self.yolo7 = YOLOv3Module(256)
        self.yolo8 = YOLOv3Module(256)
        self.yolo9 = YOLOv3Module(256)
        self.yolo10 = YOLOv3Module(256)
        self.yolo11 = YOLOv3Module(256)
        self.yolo12 = YOLOv3Module(256)
        self.yolo13 = YOLOv3Module(256)
        self.yolo14 = YOLOv3Module(256)

        self.conv15 = torch.nn.Conv2d(in_channels=256,
                                      out_channels=512,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      bias=True)

        self.yolo16 = YOLOv3Module(512)
        self.yolo17 = YOLOv3Module(512)
        self.yolo18 = YOLOv3Module(512)
        self.yolo19 = YOLOv3Module(512)
        self.yolo20 = YOLOv3Module(512)
        self.yolo21 = YOLOv3Module(512)
        self.yolo22 = YOLOv3Module(512)
        self.yolo23 = YOLOv3Module(512)

        self.conv24 = torch.nn.Conv2d(in_channels=512,
                                      out_channels=1024,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      bias=True)

        self.yolo25 = YOLOv3Module(1024)
        self.yolo26 = YOLOv3Module(1024)
        self.yolo27 = YOLOv3Module(1024)
        self.yolo28 = YOLOv3Module(1024)

        self.tconv29 = torch.nn.ConvTranspose2d(in_channels=1024,
                                                out_channels=512,
                                                kernel_size=2,
                                                stride=2,
                                                padding=0,
                                                bias=True)

        self.tconv30 = torch.nn.ConvTranspose2d(in_channels=512,
                                                out_channels=256,
                                                kernel_size=2,
                                                stride=2,
                                                padding=0,
                                                bias=True)

        self.tconv31 = torch.nn.ConvTranspose2d(in_channels=256,
                                                out_channels=128,
                                                kernel_size=2,
                                                stride=2,
                                                padding=0,
                                                bias=True)

        self.tconv32 = torch.nn.ConvTranspose2d(in_channels=128,
                                                out_channels=64,
                                                kernel_size=2,
                                                stride=2,
                                                padding=0,
                                                bias=True)

        self.tconv33 = torch.nn.ConvTranspose2d(in_channels=64,
                                                out_channels=64,
                                                kernel_size=2,
                                                stride=2,
                                                padding=0,
                                                bias=True)

        self.scores = torch.nn.ConvTranspose2d(in_channels=64,
                                               out_channels=1,
                                               kernel_size=4,
                                               stride=4,
                                               padding=0,
                                               bias=True)

        self.filter_scores = torch.nn.AvgPool2d(kernel_size=3,
                                                stride=1,
                                                padding=1)

    def forward(self, x):
        resizer = self.resizer(x)
        prior = torch.abs(
            (0.2126 * x[:, 0] + 0.7152 * x[:, 1] + 0.0722 * x[:, 2]) / 255.0 -
            1).unsqueeze(1)

        conv0 = self.conv0(resizer)
        conv0_relu = torch.nn.functional.relu(conv0, inplace=True)

        conv1 = self.conv1(conv0_relu)
        conv1_relu = torch.nn.functional.relu(conv1, inplace=True)

        yolo2 = self.yolo2(conv1_relu)

        conv3 = self.conv3(yolo2)
        conv3_relu = torch.nn.functional.relu(conv3, inplace=True)

        yolo4 = self.yolo4(conv3_relu)
        yolo5 = self.yolo5(yolo4)

        conv6 = self.conv6(yolo5)
        conv6_relu = torch.nn.functional.relu(conv6, inplace=True)

        yolo7 = self.yolo7(conv6_relu)
        yolo8 = self.yolo8(yolo7)
        yolo9 = self.yolo9(yolo8)
        yolo10 = self.yolo10(yolo9)
        yolo11 = self.yolo11(yolo10)
        yolo12 = self.yolo12(yolo11)
        yolo13 = self.yolo13(yolo12)
        yolo14 = self.yolo14(yolo13)

        conv15 = self.conv15(yolo14)
        conv15_relu = torch.nn.functional.relu(conv15, inplace=True)

        yolo16 = self.yolo16(conv15_relu)
        yolo17 = self.yolo17(yolo16)
        yolo18 = self.yolo18(yolo17)
        yolo19 = self.yolo19(yolo18)
        yolo20 = self.yolo20(yolo19)
        yolo21 = self.yolo21(yolo20)
        yolo22 = self.yolo22(yolo21)
        yolo23 = self.yolo23(yolo22)

        conv24 = self.conv24(yolo23)
        conv24_relu = torch.nn.functional.relu(conv24, inplace=True)

        yolo25 = self.yolo25(conv24_relu)
        yolo26 = self.yolo26(yolo25)
        yolo27 = self.yolo27(yolo26)
        yolo28 = self.yolo28(yolo27)

        tconv29 = self.tconv29(yolo28)
        tconv29_relu = torch.nn.functional.relu(tconv29, inplace=True)
        tconv29_residual = yolo23 - tconv29_relu
        tconv29_residual_relu = torch.nn.functional.relu(tconv29_residual,
                                                         inplace=True)

        tconv30 = self.tconv30(tconv29_residual_relu)
        tconv30_relu = torch.nn.functional.relu(tconv30, inplace=True)
        tconv30_residual = yolo14 - tconv30_relu
        tconv30_residual_relu = torch.nn.functional.relu(tconv30_residual,
                                                         inplace=True)

        tconv31 = self.tconv31(tconv30_residual_relu)
        tconv31_relu = torch.nn.functional.relu(tconv31, inplace=True)
        tconv31_residual = yolo5 - tconv31_relu
        tconv31_residual_relu = torch.nn.functional.relu(tconv31_residual,
                                                         inplace=True)

        tconv32 = self.tconv32(tconv31_residual_relu)
        tconv32_relu = torch.nn.functional.relu(tconv32, inplace=True)
        tconv32_residual = yolo2 - tconv32_relu
        tconv32_residual_relu = torch.nn.functional.relu(tconv32_residual,
                                                         inplace=True)

        tconv33 = self.tconv33(tconv32_residual_relu)
        tconv33_relu = torch.nn.functional.relu(tconv33, inplace=True)

        scores = self.scores(tconv33_relu)
        #scores = torch.tanh(scores)
        scores = torch.tanh(scores)
        scores = (scores * 0.75 + prior * 0.25)
        scores = torch.nn.functional.relu(scores, inplace=True)
        scores = self.filter_scores(scores)

        return (scores, )


class GlomerusDetector7(torch.nn.Module):
    def __init__(self):
        super(GlomerusDetector7, self).__init__()

        self.intensity = torch.nn.Conv2d(in_channels=3,
                                         out_channels=1,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         bias=False)
        self.intensity.weight.data = (torch.tensor(
            [0.2126, 0.7152, 0.0722]).unsqueeze(0).unsqueeze(2).unsqueeze(3))
        self.intensity.weight.requires_grad = False

        self.conv0 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=32,
                                     kernel_size=3,
                                     padding=1,
                                     stride=1,
                                     bias=True)

        self.conv1 = torch.nn.Conv2d(in_channels=32,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     bias=True)

        self.yolo2 = YOLOv3Module(64)

        self.conv3 = torch.nn.Conv2d(in_channels=64,
                                     out_channels=128,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     bias=True)

        self.yolo4 = YOLOv3Module(128)
        self.yolo5 = YOLOv3Module(128)

        self.conv6 = torch.nn.Conv2d(in_channels=128,
                                     out_channels=256,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     bias=True)

        self.yolo7 = YOLOv3Module(256)
        self.yolo8 = YOLOv3Module(256)
        self.yolo9 = YOLOv3Module(256)
        self.yolo10 = YOLOv3Module(256)
        self.yolo11 = YOLOv3Module(256)
        self.yolo12 = YOLOv3Module(256)
        self.yolo13 = YOLOv3Module(256)
        self.yolo14 = YOLOv3Module(256)

        self.conv15 = torch.nn.Conv2d(in_channels=256,
                                      out_channels=512,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      bias=True)

        self.yolo16 = YOLOv3Module(512)
        self.yolo17 = YOLOv3Module(512)
        self.yolo18 = YOLOv3Module(512)
        self.yolo19 = YOLOv3Module(512)
        self.yolo20 = YOLOv3Module(512)
        self.yolo21 = YOLOv3Module(512)
        self.yolo22 = YOLOv3Module(512)
        self.yolo23 = YOLOv3Module(512)

        self.conv24 = torch.nn.Conv2d(in_channels=512,
                                      out_channels=1024,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      bias=True)

        self.yolo25 = YOLOv3Module(1024)
        self.yolo26 = YOLOv3Module(1024)
        self.yolo27 = YOLOv3Module(1024)
        self.yolo28 = YOLOv3Module(1024)

        self.predictor = torch.nn.Conv2d(in_channels=1024,
                                         out_channels=32 * 17 * 2,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         bias=True)

    def forward(self, x):
        intensity = self.intensity(x)

        conv0 = self.conv0(intensity)
        conv0_relu = torch.nn.functional.relu(conv0, inplace=True)

        conv1 = self.conv1(conv0_relu)
        conv1_relu = torch.nn.functional.relu(conv1, inplace=True)

        yolo2 = self.yolo2(conv1_relu)

        conv3 = self.conv3(yolo2)
        conv3_relu = torch.nn.functional.relu(conv3, inplace=True)

        yolo4 = self.yolo4(conv3_relu)
        yolo5 = self.yolo5(yolo4)

        conv6 = self.conv6(yolo5)
        conv6_relu = torch.nn.functional.relu(conv6, inplace=True)

        yolo7 = self.yolo7(conv6_relu)
        yolo8 = self.yolo8(yolo7)
        yolo9 = self.yolo9(yolo8)
        yolo10 = self.yolo10(yolo9)
        yolo11 = self.yolo11(yolo10)
        yolo12 = self.yolo12(yolo11)
        yolo13 = self.yolo13(yolo12)
        yolo14 = self.yolo14(yolo13)

        conv15 = self.conv15(yolo14)
        conv15_relu = torch.nn.functional.relu(conv15, inplace=True)

        yolo16 = self.yolo16(conv15_relu)
        yolo17 = self.yolo17(yolo16)
        yolo18 = self.yolo18(yolo17)
        yolo19 = self.yolo19(yolo18)
        yolo20 = self.yolo20(yolo19)
        yolo21 = self.yolo21(yolo20)
        yolo22 = self.yolo22(yolo21)
        yolo23 = self.yolo23(yolo22)

        conv24 = self.conv24(yolo23)
        conv24_relu = torch.nn.functional.relu(conv24, inplace=True)

        yolo25 = self.yolo25(conv24_relu)
        yolo26 = self.yolo26(yolo25)
        yolo27 = self.yolo27(yolo26)
        yolo28 = self.yolo28(yolo27)

        pred = self.predictor(yolo28)
        pred = pred.permute((0, 2, 3, 1))
        pred = pred.reshape((pred.shape[0], pred.shape[1], pred.shape[2],
                             pred.shape[3] // 2, 2)).contiguous()
        pred = torch.view_as_complex(pred)
        pred = pred.reshape(
            (pred.shape[0], pred.shape[1], pred.shape[2], 32, 17))
        irfft_pred = torch.fft.irfftn(pred, dim=(3, 4))
        irfft_pred = irfft_pred.permute((0, 1, 3, 2, 4))
        irfft_pred = irfft_pred.reshape(
            (pred.shape[0], 1, x.shape[2], x.shape[3]))

        return (irfft_pred, )


class GlomerusDetector8(torch.nn.Module):
    def __init__(self):
        super(GlomerusDetector8, self).__init__()

        self.intensity = torch.nn.Conv2d(in_channels=3,
                                         out_channels=1,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         bias=False)
        self.intensity.weight.data = (torch.tensor(
            [0.2126, 0.7152, 0.0722]).unsqueeze(0).unsqueeze(2).unsqueeze(3))
        self.intensity.weight.requires_grad = False

        self.conv0 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=32,
                                     kernel_size=3,
                                     padding=1,
                                     stride=1,
                                     bias=True)

        self.conv1 = torch.nn.Conv2d(in_channels=32,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     bias=True)

        self.yolo2 = YOLOv3Module(64)

        self.conv3 = torch.nn.Conv2d(in_channels=64,
                                     out_channels=128,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     bias=True)

        self.yolo4 = YOLOv3Module(128)
        self.yolo5 = YOLOv3Module(128)

        self.conv6 = torch.nn.Conv2d(in_channels=128,
                                     out_channels=256,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     bias=True)

        self.yolo7 = YOLOv3Module(256)
        self.yolo8 = YOLOv3Module(256)
        self.yolo9 = YOLOv3Module(256)
        self.yolo10 = YOLOv3Module(256)
        self.yolo11 = YOLOv3Module(256)
        self.yolo12 = YOLOv3Module(256)
        self.yolo13 = YOLOv3Module(256)
        self.yolo14 = YOLOv3Module(256)

        self.conv15 = torch.nn.Conv2d(in_channels=256,
                                      out_channels=512,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      bias=True)

        self.yolo16 = YOLOv3Module(512)
        self.yolo17 = YOLOv3Module(512)
        self.yolo18 = YOLOv3Module(512)
        self.yolo19 = YOLOv3Module(512)
        self.yolo20 = YOLOv3Module(512)
        self.yolo21 = YOLOv3Module(512)
        self.yolo22 = YOLOv3Module(512)
        self.yolo23 = YOLOv3Module(512)

        self.conv24 = torch.nn.Conv2d(in_channels=512,
                                      out_channels=1024,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      bias=True)

        self.yolo25 = YOLOv3Module(1024)
        self.yolo26 = YOLOv3Module(1024)
        self.yolo27 = YOLOv3Module(1024)
        self.yolo28 = YOLOv3Module(1024)

        self.predictor = torch.nn.Conv2d(in_channels=1024,
                                         out_channels=32 * 32,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         bias=True)

    def forward(self, x):
        intensity = self.intensity(x)

        conv0 = self.conv0(intensity)
        conv0_relu = torch.nn.functional.relu(conv0, inplace=True)

        conv1 = self.conv1(conv0_relu)
        conv1_relu = torch.nn.functional.relu(conv1, inplace=True)

        yolo2 = self.yolo2(conv1_relu)

        conv3 = self.conv3(yolo2)
        conv3_relu = torch.nn.functional.relu(conv3, inplace=True)

        yolo4 = self.yolo4(conv3_relu)
        yolo5 = self.yolo5(yolo4)

        conv6 = self.conv6(yolo5)
        conv6_relu = torch.nn.functional.relu(conv6, inplace=True)

        yolo7 = self.yolo7(conv6_relu)
        yolo8 = self.yolo8(yolo7)
        yolo9 = self.yolo9(yolo8)
        yolo10 = self.yolo10(yolo9)
        yolo11 = self.yolo11(yolo10)
        yolo12 = self.yolo12(yolo11)
        yolo13 = self.yolo13(yolo12)
        yolo14 = self.yolo14(yolo13)

        conv15 = self.conv15(yolo14)
        conv15_relu = torch.nn.functional.relu(conv15, inplace=True)

        yolo16 = self.yolo16(conv15_relu)
        yolo17 = self.yolo17(yolo16)
        yolo18 = self.yolo18(yolo17)
        yolo19 = self.yolo19(yolo18)
        yolo20 = self.yolo20(yolo19)
        yolo21 = self.yolo21(yolo20)
        yolo22 = self.yolo22(yolo21)
        yolo23 = self.yolo23(yolo22)

        conv24 = self.conv24(yolo23)
        conv24_relu = torch.nn.functional.relu(conv24, inplace=True)

        yolo25 = self.yolo25(conv24_relu)
        yolo26 = self.yolo26(yolo25)
        yolo27 = self.yolo27(yolo26)
        yolo28 = self.yolo28(yolo27)

        #scores = torch.nn.functional.relu(torch.tanh(self.predictor(yolo28)))
        scores = torch.sigmoid(self.predictor(yolo28))
        #print(f"b: {torch.cuda.memory_stats()['active_bytes.all.current'] / 1024 / 1024}")
        scores = scores.permute((0, 2, 3, 1))
        scores = scores.reshape(
            (scores.shape[0], scores.shape[1], scores.shape[2], 32, 32))
        scores = scores.permute((0, 1, 3, 2, 4))
        scores = scores.reshape((scores.shape[0], 1, x.shape[2], x.shape[3]))
        #print(f"a: {torch.cuda.memory_stats()['active_bytes.all.current'] / 1024 / 1024}")

        return (scores, )

class GlomerusDetector9(torch.nn.Module):
    def __init__(self):
        super(GlomerusDetector9, self).__init__()

        self.resizer = torch.nn.Conv2d(in_channels=3,
                                       out_channels=1,
                                       kernel_size=9,
                                       stride=4,
                                       padding=4,
                                       bias=False)

        self.conv0 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=32,
                                     kernel_size=3,
                                     padding=1,
                                     stride=1,
                                     bias=True)

        self.conv1 = torch.nn.Conv2d(in_channels=32,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     bias=True)

        self.yolo2 = YOLOv3Module(64)

        self.conv3 = torch.nn.Conv2d(in_channels=64,
                                     out_channels=128,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     bias=True)

        self.yolo4 = YOLOv3Module(128)
        self.yolo5 = YOLOv3Module(128)

        self.conv6 = torch.nn.Conv2d(in_channels=128,
                                     out_channels=256,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     bias=True)

        self.yolo7 = YOLOv3Module(256)
        self.yolo8 = YOLOv3Module(256)
        self.yolo9 = YOLOv3Module(256)
        #self.yolo10 = YOLOv3Module(256)
        #self.yolo11 = YOLOv3Module(256)
        #self.yolo12 = YOLOv3Module(256)
        #self.yolo13 = YOLOv3Module(256)
        #self.yolo14 = YOLOv3Module(256)

        self.conv15 = torch.nn.Conv2d(in_channels=256,
                                      out_channels=512,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      bias=True)

        self.yolo16 = YOLOv3Module(512)
        self.yolo17 = YOLOv3Module(512)
        self.yolo18 = YOLOv3Module(512)
        #self.yolo19 = YOLOv3Module(512)
        #self.yolo20 = YOLOv3Module(512)
        #self.yolo21 = YOLOv3Module(512)
        #self.yolo22 = YOLOv3Module(512)
        #self.yolo23 = YOLOv3Module(512)

        self.conv24 = torch.nn.Conv2d(in_channels=512,
                                      out_channels=1024,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      bias=True)

        self.yolo25 = YOLOv3Module(1024)
        self.yolo26 = YOLOv3Module(1024)
        #self.yolo27 = YOLOv3Module(1024)
        #self.yolo28 = YOLOv3Module(1024)

        self.tconv29 = torch.nn.ConvTranspose2d(in_channels=1024,
                                                out_channels=512,
                                                kernel_size=2,
                                                stride=2,
                                                padding=0,
                                                bias=True)

        self.tconv30 = torch.nn.ConvTranspose2d(in_channels=512,
                                                out_channels=256,
                                                kernel_size=2,
                                                stride=2,
                                                padding=0,
                                                bias=True)

        self.tconv31 = torch.nn.ConvTranspose2d(in_channels=256,
                                                out_channels=128,
                                                kernel_size=2,
                                                stride=2,
                                                padding=0,
                                                bias=True)

        self.tconv32 = torch.nn.ConvTranspose2d(in_channels=128,
                                                out_channels=64,
                                                kernel_size=2,
                                                stride=2,
                                                padding=0,
                                                bias=True)

        self.tconv33 = torch.nn.ConvTranspose2d(in_channels=64,
                                                out_channels=64,
                                                kernel_size=2,
                                                stride=2,
                                                padding=0,
                                                bias=True)

        self.scores = torch.nn.ConvTranspose2d(in_channels=64,
                                               out_channels=1,
                                               kernel_size=4,
                                               stride=4,
                                               padding=0,
                                               bias=True)


    def forward(self, x):
        resizer = self.resizer(x)

        conv0 = self.conv0(resizer)
        conv0_relu = torch.nn.functional.relu(conv0, inplace=True)

        conv1 = self.conv1(conv0_relu)
        conv1_relu = torch.nn.functional.relu(conv1, inplace=True)

        yolo2 = self.yolo2(conv1_relu)

        conv3 = self.conv3(yolo2)
        conv3_relu = torch.nn.functional.relu(conv3, inplace=True)

        yolo4 = self.yolo4(conv3_relu)
        yolo5 = self.yolo5(yolo4)

        conv6 = self.conv6(yolo5)
        conv6_relu = torch.nn.functional.relu(conv6, inplace=True)

        yolo7 = self.yolo7(conv6_relu)
        yolo8 = self.yolo8(yolo7)
        yolo9 = self.yolo9(yolo8)
        #yolo10 = self.yolo10(yolo9)
        #yolo11 = self.yolo11(yolo10)
        #yolo12 = self.yolo12(yolo11)
        #yolo13 = self.yolo13(yolo12)
        #yolo14 = self.yolo14(yolo13)
        yolo14 = yolo9

        conv15 = self.conv15(yolo14)
        conv15_relu = torch.nn.functional.relu(conv15, inplace=True)

        yolo16 = self.yolo16(conv15_relu)
        yolo17 = self.yolo17(yolo16)
        yolo18 = self.yolo18(yolo17)
        #yolo19 = self.yolo19(yolo18)
        #yolo20 = self.yolo20(yolo19)
        #yolo21 = self.yolo21(yolo20)
        #yolo22 = self.yolo22(yolo21)
        #yolo23 = self.yolo23(yolo22)
        yolo23 = yolo18

        conv24 = self.conv24(yolo23)
        conv24_relu = torch.nn.functional.relu(conv24, inplace=True)

        yolo25 = self.yolo25(conv24_relu)
        yolo26 = self.yolo26(yolo25)
        #yolo27 = self.yolo27(yolo26)
        #yolo28 = self.yolo28(yolo27)
        yolo28 = yolo26

        tconv29 = self.tconv29(yolo28)
        tconv29_relu = torch.nn.functional.relu(tconv29, inplace=True)
        tconv29_residual = yolo23 - tconv29_relu
        tconv29_residual_relu = torch.nn.functional.relu(tconv29_residual,
                                                         inplace=True)

        tconv30 = self.tconv30(tconv29_residual_relu)
        tconv30_relu = torch.nn.functional.relu(tconv30, inplace=True)
        tconv30_residual = yolo14 - tconv30_relu
        tconv30_residual_relu = torch.nn.functional.relu(tconv30_residual,
                                                         inplace=True)

        tconv31 = self.tconv31(tconv30_residual_relu)
        tconv31_relu = torch.nn.functional.relu(tconv31, inplace=True)
        tconv31_residual = yolo5 - tconv31_relu
        tconv31_residual_relu = torch.nn.functional.relu(tconv31_residual,
                                                         inplace=True)

        tconv32 = self.tconv32(tconv31_residual_relu)
        tconv32_relu = torch.nn.functional.relu(tconv32, inplace=True)
        tconv32_residual = yolo2 - tconv32_relu
        tconv32_residual_relu = torch.nn.functional.relu(tconv32_residual,
                                                         inplace=True)

        tconv33 = self.tconv33(tconv32_residual_relu)
        tconv33_relu = torch.nn.functional.relu(tconv33, inplace=True)

        scores = self.scores(tconv33_relu)
        scores = torch.sigmoid(scores)

        return (scores, )
