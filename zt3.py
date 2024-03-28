import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F


#这个类的主要作用是执行卷积操作。这个卷积操作可以通过输入数据上滑动卷积核(滤波器)完成，这个过程可提取出输入数据的局部特征。
class Conv2d(nn.Module):
    #多个参数，包括输入通道数（in_channels）、输出通道数（out_channels）、卷积核大小（kernel_size）、步长（stride）、扩张率（dilation）。参数定义了卷积层的行为和输出的特征图的尺寸。
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bn=False, se='relu', pad=True):
        super(Conv2d, self).__init__()

        #如果 bn 参数为 True，则在卷积层后添加一个 nn.BatchNorm2d 层。
        # print('I am in network conv2d.init')
        padding = int(dilation * (kernel_size - 1) / 2) if pad == True else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation=dilation, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None

        #根据参数 se 的值选择不同的激活函数，如 ReLU 或 Sigmoid。学习模拟更复杂的函数
        if se == "relu":
            self.relu = nn.ReLU(inplace=True)
        elif se == "sigmoid":
            self.relu = nn.Sigmoid()
        else:
            self.relu = None
        # self.sigmoid = nn.ReLU(inplace=True) if sigmoid else None


    #forward方法中，输入数据x先通过卷积层，之后需要经过批量归一化层和激活函数返回处理之后的数据。
    def forward(self, x):
        # print("iam in work conv2d forward")
        x = self.conv(x)
        if self.bn is not None:
            '''
            如果bn参数为True,则会在卷积层后添加批量归一化（Batch Normalization）层
            '''
            x = self.bn(x)
        if self.relu is not None:
            '''
            同样，这里也会引进激活函数
            '''
            x = self.relu(x)

        return x


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    '''
    函数根据配置列表动态构建一个序列化层结构，这个列表可以包含卷积层和池化层的配置。
    '''
    '''
    使用 make_layers 函数根据 Conv3_3f、Conv4_3f 和 Conv5_3f 配置列表构建卷积层序列。这些函数调用会创建 nn.Sequential 对象，其中包含了卷积层和批量归一化层（如果启用）。
    '''
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':

            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class ADConv(nn.Module):
    '''
    这是一个自适应卷积层，它使用一个卷积层来生成注意力权重，这个层通过卷积操作和 Softmax 函数来生成一个注意力分布，这个分布决定了哪些位置的特征更重要。
    '''

    def __init__(self, p_num):
        super(ADConv, self).__init__()
        self.conv = nn.Conv2d(1, 1, 1)
        self.act = nn.Softmax(dim=-1)
        self.pos_mask = nn.Sequential(
            nn.Linear(p_num, 2 * p_num, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2 * p_num, p_num, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        forward方法中，对输入的图像进行一系列的卷积操作。
        '''

        # print("begin", torch.max(x), torch.min(x))
        B, L, num = x.shape
        x = x.view(B, 1, L, num)
        x = self.conv(x).squeeze(dim=1)
        device = x.get_device()
        # one = torch.ones_like(x, device=device) * (-100000)
        # zero = torch.zeros_like(x, device=device)
        one = torch.ones_like(x) * (-100000)
        zero = torch.zeros_like(x)
        mask = self.pos_mask(torch.where(x > 0, x, zero))
        x = self.act(torch.where(x > 0, x, one))

        return x * mask


'''
注意力机制，增强对图像中重要特征的关注能力，帮助更加关注于图像关键部分。权重注意力机制部分代码
'''
class multi_att(nn.Module):
    def __init__(self, dim, top=9, c_ratio=8):
        super(multi_att, self).__init__()
        self.p_num = top
        cim = int(dim[0] / c_ratio)
        self.proj_q = nn.Linear(dim[0], cim, bias=False)
        self.proj_k = nn.Linear(dim[0], cim, bias=False)
        self.proj_v = nn.Linear(dim[0], cim, bias=False)
        '''
        投影层
        proj_q、proj_k 和 proj_v 是三个线性层（nn.Linear），它们分别用于将输入特征投影到一个新的空间，这个空间的维度由 cim（计算输入维度除以一个压缩比 c_ratio 得到）决定。这些投影层对应于注意力机制中的查询（query）、键（key）和值（value）。
        '''
        self.adptive = ADConv(top)
        #adptive是一个自适应卷积层，用于生成注意力权重
        self.catt = nn.Parameter(torch.ones([1, dim[0], 1, 1]), requires_grad=True)
        self.catt1 = nn.Parameter(torch.ones([1, dim[0], 1, 1]), requires_grad=True)
        self.dsn = Conv2d(dim[0], 1, 1, bn=True)
        self.back = nn.Linear(cim, dim[0], bias=False)
        self.mlp = Conv2d(dim[0] * 2, dim[0], 1, bn=True)

    def forward(self, feat, H, W):
        '''
        前向传播函数
        '''
        '''
        :param feat: list with different scale feature
        :param size:
        :return:
        '''
        # print ("Train",Train)
        #
        #根据输入图像的不同大小决定调用哪一个函数
        if H < 768 and W < 768:
            # get_forward用于处理尺寸较小的特征图
            return self.get_forward(feat)
        else:
            # get_split_forward用于处理尺寸较大的特征图
            return self.get_split_forward(feat)

   #get_forward用于处理尺寸较小的特征图
    def get_forward(self, feat):
        '''
        这个函数主要用于权重注意力计算
        :param feat:
        :return:
        '''
        #q、k、v矩阵分别计算出来
        B, C, H, W = feat[0].shape
        q = self.proj_q(feat[0].flatten(2).transpose(1, 2))  # b,h'w',c/4
        k = self.proj_k(feat[0].flatten(2).transpose(1, 2))
        v = self.proj_v(feat[0].flatten(2).transpose(1, 2))

        #w为q、k计算的余弦相似度
        w = cos_dot(q, k)  # (b,hw,c/4) (b,hw1*4,c/4) b,hw,hw1*4
        del q, k

        #torch.topk主要选择top-k中最重要的特证
        w, index = torch.topk(w, self.p_num, dim=-1)  # b,hw,n0um

        #通过 get_top_value 函数从 value 中提取相应的特征，之后通过adptive 层进行加权，用于构建最终的输出特征图。
        v = get_top_value(v, index)  # (b,hw1*3,c) (b,hw,num) -> b,hw,num,c
        w = self.adptive(w)
        w = w.view(B, H * W, 1, self.p_num)  # b,HW,1,num
        v = torch.matmul(w, v).squeeze(dim=2)  # b,hw,c

        #通过back线性层将加权后的特征转换回原始维度。
        v = self.back(v)
        v = self.catt * v.transpose(1, 2).reshape(B, C, H, W)

        #使用mlp(多层感知机)和dsn(下采样层)进一步处理融合特征
        x = self.mlp(torch.cat([v, self.catt1 * feat[0]], dim=1))
        r0 = self.dsn(x)
        return x, r0

    #get_split_forward函数处理计算注意力权重的方法与get_forward函数中的方法一致
    def get_split_forward(self, feat):
        '''
        :param feat: list with different scale feature
        :param size:
        :return:
        '''

        # print ("Train",Train)
        # s_feat=feat[0]
        B, C, H, W = feat[0].shape
        if H >= 384:
            hn = 4
        elif H >= 192:
            hn = 2
        else:
            hn = 1
        if W >= 384:
            wn = 4
        elif W >= 192:
            wn = 2
        else:
            wn = 1

        f = get_chunck(feat[0], [hn, wn])

        #分别计算q、k、v矩阵
        q = self.proj_q(f.flatten(2).transpose(1, 2))  # b,hw1,c/4
        k = self.proj_k(f.flatten(2).transpose(1, 2))
        v = self.proj_v(f.flatten(2).transpose(1, 2))

        #计算余弦相似度
        w = cos_dot(q, k)
        del q, k

        #使用 torch.topk 函数选择 top-k 最重要的特征
        w, index = torch.topk(w, self.p_num, dim=-1)  # b,hw,n0um

        #通过 get_top_value 函数从 value 中提取相应的特征。
        v = get_top_value(v, index)  # (b,hw1*3,c) (b,hw,num) -> b,hw,num,c
        w = self.adptive(w)
        w = w.view(-1, int(H / hn) * int(W / wn), 1, self.p_num)  # b,HW,1,num
        v = torch.matmul(w, v).squeeze(dim=2)  # b,hw,c
        v = self.back(v).transpose(1, 2).reshape(-1, C, int(H / hn), int(W / wn))
        v = get_back(v, [hn, wn])
        v = self.catt * v
        x = self.mlp(torch.cat([v, self.catt1 * feat[0]], dim=1))
        r3 = self.dsn(x)
        return x, r3


class zt3(nn.Module):
    '''
    在 zt3 类的构造函数中，定义了模型的不同层，包括卷积层、池化层和注意力模块。
   使用 make_layers 函数根据配置构建卷积层序列。
   定义了三个不同的卷积层序列 Conv3_3、Conv4_3 和 Conv5_3，它们分别对应不同的特征图尺度。
   定义了三个转换层 T1、T2 和 T3，以及一个用于生成最终输出的层。
   定义了一些特定的卷积层用于生成注意力权重和进行特征增强。
   如果不加载预训练权重，模型会使用 VGG16 预训练模型的权重进行初始化。
    '''

    def __init__(self, load_weights=False):
        #先定义类属性
        '''
        Conv3_3f、Conv4_3f 和 Conv5_3f 是模型中不同层的特征通道数的列表，分别对应三个不同阶段的卷积层配置。
        :param load_weights:
        '''
        super(zt3, self).__init__()
        self.Conv3_3f = [64, 64, 'M', 128, 128, 'M', 256, 256, 256]
        self.Conv4_3f = ['M', 512, 512, 512]
        self.Conv5_3f = ['M', 512, 512, 512]

        '''
        使用 make_layers 函数根据 Conv3_3f、Conv4_3f 和 Conv5_3f 配置列表构建卷积层序列。这些函数调用会创建 nn.Sequential 对象，其中包含了卷积层和批量归一化层（如果启用）。
        '''
        # self.Conv2_2 = make_layers(self.Conv2_2f, in_channels=3, batch_norm=True, dilation=False)
        self.Conv3_3 = make_layers(self.Conv3_3f, in_channels=3, batch_norm=True, dilation=False)
        self.Conv4_3 = make_layers(self.Conv4_3f, in_channels=256, batch_norm=True, dilation=False)
        self.Conv5_3 = make_layers(self.Conv5_3f, in_channels=512, batch_norm=True, dilation=False)
        # self.back = make_layers(self.Conv5_3f, in_channels=512, batch_norm=True, dilation=False)


        # T1、T2 和 T3 是模型中的三个不同的转换序列（transition layers），它们负责在不同层之间转换特征图。
        #
        self.T1 = nn.Sequential(
            Conv2d(1025, 256, 1, bn=True),
            Conv2d(256, 256, 3, bn=True),
        )

        self.T2 = nn.Sequential(
            Conv2d(513, 128, 1, bn=True),
            Conv2d(128, 128, 3, bn=True),
        )

        self.T3 = nn.Sequential(
            Conv2d(128, 64, 3, bn=True),
            Conv2d(64, 64, 3, bn=True),
            Conv2d(64, 1, 1, bn=True),
        )

        self.dmnT3 = Conv2d(128, 1, 1, bn=True, se='sigmoid')
        self.d1024b = Conv2d(1024, 1, 1, bn=True)
        self.d512b = Conv2d(512, 1, 1, bn=True)
        self.enhance_pos = multi_att(dim=[512], top=6)

        #初始化权重
        '''
        如果 load_weights 参数为 False，则模型将使用预训练的 VGG16 模型的权重进行初始化。这是通过加载 VGG16 模型的权重到 zt3 模型的对应层中实现的。
        '''
        if not load_weights:
            mod = models.vgg16_bn(pretrained=True)
            self._initialize_weights()
            '''
            如果 load_weights 为 False，则调用 _initialize_weights 方法来初始化模型中所有 Conv2d 和 Linear 层的权重。这通常使用 Xavier 初始化方法完成。
            '''
            for j in range(len(self.Conv3_3)):
                self.Conv3_3[j].load_state_dict(mod.features[j].state_dict())
            for p in range(len(self.Conv4_3)):
                self.Conv4_3[p].load_state_dict(mod.features[j + p + 1].state_dict())
            for q in range(len(self.Conv5_3)):
                # self.Conv5_3[q].load_state_dict(mod.features[i + j + p + q + 3+ 1].state_dict())
                self.Conv5_3[q].load_state_dict(mod.features[j + p + q + 2].state_dict())

    def forward(self, img):  # the shape of x is 3,3,368,640
        '''
        zt3 类的 forward 方法定义了数据通过网络的流程。输入图像首先通过一系列卷积层和池化层，生成不同尺度的特征图。
        '''
        B, C, H, W = img.shape
        c3 = self.Conv3_3(img)
        c4 = self.Conv4_3(c3)
        c5 = self.Conv5_3(c4)
        c5, r0 = self.enhance_pos([c5], H, W)
        s1 = F.interpolate(c5, scale_factor=2, mode='bilinear')

        s1 = torch.cat((s1, c4), 1)
        del c4
        r1 = self.d1024b(s1)

        s1 = self.T1(torch.cat((s1, r1), 1))
        s2 = F.interpolate(s1, scale_factor=2, mode='bilinear')
        s2 = torch.cat((s2, c3), 1)
        del c3
        r2 = self.d512b(s2)
        s2 = self.T2(torch.cat((s2, r2), 1))
        mask = self.dmnT3(s2)
        s2 = s2 * mask
        r4 = self.T3(s2)
        return r4, [r2, r1, r0], mask

    #这个方法的作用是初始化模型中的权重，
    def _initialize_weights(self):
        # self.modules() 函数遍历模型中的所有模块
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #如果它是一个 nn.Conv2d 类型的层（即二维卷积层），则使用 Xavier（或 Glorot）初始化方法来初始化该层的权重。这个权重使用nn.init.xavier_uniform_ 函数进行初始化，此函数会生成一个均匀分布的随机数，范围是根据层的输入和输出维度自动计算的。
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                #nn.Linear 类型的层（即全连接层），也使用 Xavier 初始化方法来初始化权重，这样可以有助于在全连接层中保持激活函数和梯度分布的稳定。
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)




'''
get_back函数接收两个参数，img是经过注意力机制处理的特征图，可能被分割为多个块便于计算。split_r表示图像在高度和宽度方向的分割比例。
'''
def get_back(img, split_r):
    #根据 split_r 中的值将 img 分割成多个块，在高度和宽度范围进行分割小块，每个块对应于原始图像中的一个子区域。
    B, C, H, W = img.shape
    hn, wn = split_r
    #分割之后，再将这些块在相反的方向重新拼接起来，进而恢复出完整的特征图
    img = torch.cat(torch.split(img, wn, dim=0), dim=2)
    img = torch.cat(torch.split(img, 1, dim=0), dim=-1)
    #最终，get_back 函数返回重组后的特征图，其空间分辨率与原始输入图像相匹配。
    return img


#get_chunk接收两个参数，input是一个四维张量，表示为：批量大小*通道数*高度*宽度。size是一个元组，包含两个元素，分别表示高度和宽度方向上进行分割的块数。
def get_chunck(input, size):
    # input = torch.randn(1,3,8,6)
    # print ('ori',input)
    B, C, H, W = input.shape
    img = []
    #使用torch.split对高度和宽度方向的输入变量进行分割，并且在指定维度将张量分割为多个子张量
    row = torch.split(input, int(H / size[0]), dim=2)

    #函数将这些子张量重新组织成一个新的张量。在高度方向上，它将列子张量（垂直分割）拼接成一个新的张量；然后，在宽度方向上，它将行子张量（水平分割）拼接成最终的张量。
    for r in row:
        img += [torch.cat(torch.split(r, int(W / size[1]), dim=-1), dim=0)]
    img = torch.cat(img, dim=0)
    #函数返回重组之后的张量，包含所有分割的块，并在一个新的维度上被拼凑，四维张量变为三维张量。
    return img


#get_top_value 函数在多尺度注意力机制中的作用是根据计算得到的注意力权重，从一组特征中选择每个位置的 top-k 最重要的特征值。
def get_top_value(value, index):
    '''
    value: b, pri1, other. Where, pri1 reprensets the dim need be selceted, and other is the other dim
    index:b,pri,num. Where num is response to pri1 in value. here index means the i-th feat in pri need nums in pri1
    '''
    b, pri1, other = value.shape
    b, pri, num = index.shape
    indext = torch.stack([index[i] + i * pri1 for i in range(b)], dim=0)
    value = value.view(-1, other)  # Bc1,hw
    indext = indext.view(-1, num)
    indext = indext.transpose(0, 1).contiguous()  # num,bpri
    value = torch.stack([torch.index_select(value, dim=0, index=n) for n in indext], dim=1)
    value = value.view(b, pri, num, -1)

    #返回重塑后的特证张量，包含根据注意力分数选择的top-k最重要特征
    return value

#计算top-k权重。(cos值)
def cos_dot(x, y):
    '''
    x:b,c,hw / b,hw,c
    y:b,c1,hw / b,hw,c
    output:
    weight:b,c,c1 cos_smilarity
    '''
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    y = y.transpose(1, 2).contiguous()
    weight = torch.matmul(x, y)  # b,c,c1
    return weight


def get_mask(feat, mask):
    B, C, H, W = feat.shape
    mask = F.interpolate(mask, size=[H, W], mode='bilinear', align_corners=None)
    return feat * mask

'''
创建了一个 zt3 模型实例。
使用一个全为1的张量作为输入图像，模拟模型的前向传播过程。
打印出模型输出的大小，这通常用于验证模型是否正确构建和执行。
'''

if __name__ == '__main__':
    ##1.创建zt3实例化对象
    model = zt3()
    # x = torch.ones(1, 3, 256, 256)
    ##2.准备输入数据
    x = torch.ones(1, 3, 256, 256)
    ##3.将输入数据 x 传递给模型 model，执行前向传播过程。
    (mu,mu_norm) = model(x)
    ##4.打印输出结果 mu 和 mu_norm 的大小。
    print(mu.size(), mu_norm.size())
