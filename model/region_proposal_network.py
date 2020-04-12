import numpy as np
from torch.nn import functional as F
import torch as t
from torch import nn

from model.utils.bbox_tools import generate_anchor_base
from model.utils.creator_tool import ProposalCreator


class RegionProposalNetwork(nn.Module):
    """Region Proposal Network introduced in Faster R-CNN.

    This is Region Proposal Network introduced in Faster R-CNN [#]_.
    This takes features extracted from images and propose
    class agnostic bounding boxes around "objects".

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        in_channels (int): The channel size of input.
        mid_channels (int): The channel size of the intermediate tensor.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.
        feat_stride (int): Stride size after extracting features from an
            image.
        initialW (callable): Initial weight value. If :obj:`None` then this
            function uses Gaussian distribution scaled by 0.1 to
            initialize weight.
            May also be a callable that takes an array and edits its values.
        proposal_creator_params (dict): Key valued paramters for
            :class:`model.utils.creator_tools.ProposalCreator`.

    .. seealso::
        :class:`~model.utils.creator_tools.ProposalCreator`

    """

    def __init__(
            self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16,
            proposal_creator_params=dict(),
    ):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base( #生成anchor_base
            anchor_scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride #下采样倍数
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params) #实例化生成roi函数
        n_anchor = self.anchor_base.shape[0] #anchor_base的数量
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1) #3x3卷积核
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0) #rpn分类层
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0) #rpn回归层
        normal_init(self.conv1, 0, 0.01) #初始化3x3卷积核参数
        normal_init(self.score, 0, 0.01) #初始化rpn分类层参数
        normal_init(self.loc, 0, 0.01) #初始化rpn回归层参数

    def forward(self, x, img_size, scale=1.):
        """Forward Region Proposal Network.

        Here are notations.

        * :math:`N` is batch size.
        * :math:`C` channel size of the input.
        * :math:`H` and :math:`W` are height and witdh of the input feature.
        * :math:`A` is number of anchors assigned to each pixel.

        Args:
            x (~torch.autograd.Variable): The Features extracted from images.
                Its shape is :math:`(N, C, H, W)`.
            img_size (tuple of ints): A tuple :obj:`height, width`,
                which contains image size after scaling.
            scale (float): The amount of scaling done to the input images after
                reading them from files.

        Returns:
            (~torch.autograd.Variable, ~torch.autograd.Variable, array, array, array):

            This is a tuple of five following values.

            * **rpn_locs**: Predicted bounding box offsets and scales for \
                anchors. Its shape is :math:`(N, H W A, 4)`.
            * **rpn_scores**:  Predicted foreground scores for \
                anchors. Its shape is :math:`(N, H W A, 2)`.
            * **rois**: A bounding box array containing coordinates of \
                proposal boxes.  This is a concatenation of bounding box \
                arrays from multiple images in the batch. \
                Its shape is :math:`(R', 4)`. Given :math:`R_i` predicted \
                bounding boxes from the :math:`i` th image, \
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            * **roi_indices**: An array containing indices of images to \
                which RoIs correspond to. Its shape is :math:`(R',)`.
            * **anchor**: Coordinates of enumerated shifted anchors. \
                Its shape is :math:`(H W A, 4)`.

        """
        n, _, hh, ww = x.shape
        anchor = _enumerate_shifted_anchor( #生成feature map的anchor
            np.array(self.anchor_base),
            self.feat_stride, hh, ww)

        n_anchor = anchor.shape[0] // (hh * ww) #feature map的anchor数量
        h = F.relu(self.conv1(x)) #经过3x3卷积核

        rpn_locs = self.loc(h) #回归层，shape=(n,num_anchor*4,h,w)
        # UNNOTE: check whether need contiguous
        # A: Yes
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        rpn_scores = self.score(h) #分类层,shape=(n,num_anchor*2,h,w)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4) #softmax得到分类概率
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous() #前景的概率,shape=(n,hh,ww,n_anchor,)
        rpn_fg_scores = rpn_fg_scores.view(n, -1) #图片所有anchor的前景概率
        rpn_scores = rpn_scores.view(n, -1, 2) #图片所有anchor的前背景概率

        rois = list() #保存roi的list
        roi_indices = list() #保存roi对应哪张图片的list
        for i in range(n):
            #rpn_locs预测框的偏移值(anchor和rpn_locs生成roi),shape=(R,4)
            #rpn_fg_scores预测框的前景概率(根据排序前景分数从大到小获取前12000个roi),(R,)
            #anchor锚框(生成roi用),(R, 4)
            #img_size这张图片的size(用于限制超出图片的框)
            #scale图片预处理缩放后与原图的缩放比例(某种筛选作用,暂时还不清楚)
            '''1.生成roi(anchor+rpn_locs)
            2.限制超出图片的roi(roi+img_size)
            3.用scale筛选roi(scale)
            4.根据前景分数筛选12000个roi(rpn_fg_scores)
            5.根据nms控制roi数量
            6.得到shape=(S,4)的roi'''
            roi = self.proposal_layer(
                #这部分的操作不需要进行反向传播，因此可以利用numpy/tensor实现。
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32) #roi对应的图片,(S)
            rois.append(roi) #保存roi
            roi_indices.append(batch_index) #保存roi对应该图片
            #rois.shape = (n,S,4), roi_indices.shape=(n,S)

        rois = np.concatenate(rois, axis=0) #转换为np.ndarray类型，shape=(n,S,4)
        roi_indices = np.concatenate(roi_indices, axis=0) #转换为np.ndarray类型，shape=(n,S)
        #最后返回rpn自己的输出rpn_locs, rpn_scores以及给下面继续使用的输出rois, roi_indices, anchor
        return rpn_locs, rpn_scores, rois, roi_indices, anchor

#获得feature_map的anchor
def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    # it seems that it can't be boosed using GPU
    import numpy as xp
    shift_y = xp.arange(0, height * feat_stride, feat_stride) #height,一列的坐标,shape=(0,height)
    shift_x = xp.arange(0, width * feat_stride, feat_stride) #width,一行的坐标,shape=(0,width)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y) #这个操作类似笛卡尔积，用行列坐标拼凑出一张图,shift_x是每个点的横坐标shape=(height,width),shift_y是每个点的纵坐标shape=(hegith,width)
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(), #每个点的(y,x,y,x),因为bbox坐标是(ymin,xmin,ymax,xmax),不是(xmin,ymin,xmax,ymax)
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0] #K = height*width
    anchor = anchor_base.reshape((1, A, 4)) + \ #组合,shape=(K,A,4)
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32) 
    return anchor


def _enumerate_shifted_anchor_torch(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    import torch as t
    shift_y = t.arange(0, height * feat_stride, feat_stride)
    shift_x = t.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


def normal_init(m, mean, stddev, truncated=False): #初始化网络参数
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
