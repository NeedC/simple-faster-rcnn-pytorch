from __future__ import  absolute_import
from __future__ import division
import torch as t
import numpy as np
import cupy as cp
from utils import array_tool as at
from model.utils.bbox_tools import loc2bbox
from model.utils.nms import non_maximum_suppression

from torch import nn
from data.dataset import preprocess
from torch.nn import functional as F
from utils.config import opt


def nograd(f): #不计算梯度函数
    def new_f(*args,**kwargs):
        with t.no_grad():
           return f(*args,**kwargs)
    return new_f

class FasterRCNN(nn.Module): #定义FasterRCNN架构类
    """Base class for Faster R-CNN.

    This is a base class for Faster R-CNN links supporting object detection
    API [#]_. The following three stages constitute Faster R-CNN.

    1. **Feature extraction**: Images are taken and their \
        feature maps are calculated.
    2. **Region Proposal Networks**: Given the feature maps calculated in \
        the previous stage, produce set of RoIs around objects.
    3. **Localization and Classification Heads**: Using feature maps that \
        belong to the proposed RoIs, classify the categories of the objects \
        in the RoIs and improve localizations.

    Each stage is carried out by one of the callable
    :class:`torch.nn.Module` objects :obj:`feature`, :obj:`rpn` and :obj:`head`.

    There are two functions :meth:`predict` and :meth:`__call__` to conduct
    object detection.
    :meth:`predict` takes images and returns bounding boxes that are converted
    to image coordinates. This will be useful for a scenario when
    Faster R-CNN is treated as a black box function, for instance.
    :meth:`__call__` is provided for a scnerario when intermediate outputs
    are needed, for instance, for training and debugging.

    Links that support obejct detection API have method :meth:`predict` with
    the same interface. Please refer to :meth:`predict` for
    further details.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        extractor (nn.Module): A module that takes a BCHW image
            array and returns feature maps.
        rpn (nn.Module): A module that has the same interface as
            :class:`model.region_proposal_network.RegionProposalNetwork`.
            Please refer to the documentation found there.
        head (nn.Module): A module that takes
            a BCHW variable, RoIs and batch indices for RoIs. This returns class
            dependent localization paramters and class scores.
        loc_normalize_mean (tuple of four floats): Mean values of
            localization estimates.
        loc_normalize_std (tupler of four floats): Standard deviation
            of localization estimates.

    """

    def __init__(self, extractor, rpn, head,
                loc_normalize_mean = (0., 0., 0., 0.),
                loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
    ):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor #特征提取网络
        self.rpn = rpn #rpn网络
        self.head = head #roi pooling + 全连接层网络

        # mean and std
        self.loc_normalize_mean = loc_normalize_mean #坐标归一化参数
        self.loc_normalize_std = loc_normalize_std #坐标归一化参数
        self.use_preset('evaluate') #可视化内容，（跳过）

    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.head.n_class

    def forward(self, x, scale=1.):
        """Forward Faster R-CNN.

        Scaling paramter :obj:`scale` is used by RPN to determine the
        threshold to select small objects, which are going to be
        rejected irrespective of their confidence scores.

        Here are notations used.

        * :math:`N` is the number of batch size
        * :math:`R'` is the total number of RoIs produced across batches. \
            Given :math:`R_i` proposed RoIs from the :math:`i` th image, \
            :math:`R' = \\sum _{i=1} ^ N R_i`.
        * :math:`L` is the number of classes excluding the background.

        Classes are ordered by the background, the first class, ..., and
        the :math:`L` th class.

        Args:
            x (autograd.Variable): 4D image variable.
            scale (float): Amount of scaling applied to the raw image
                during preprocessing.

        Returns:
            Variable, Variable, array, array:
            Returns tuple of four values listed below.

            * **roi_cls_locs**: Offsets and scalings for the proposed RoIs. \
                Its shape is :math:`(R', (L + 1) \\times 4)`.
            * **roi_scores**: Class predictions for the proposed RoIs. \
                Its shape is :math:`(R', L + 1)`.
            * **rois**: RoIs proposed by RPN. Its shape is \
                :math:`(R', 4)`.
            * **roi_indices**: Batch indices of RoIs. Its shape is \
                :math:`(R',)`.

        """
        img_size = x.shape[2:] #得到图片尺寸(H,W)

        h = self.extractor(x) #特征提取
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \ #rpn网络，得到rpn的两个输出，rois，roi_indices，以及anchor。两个输出和anchor可用于训练
            self.rpn(h, img_size, scale)
        roi_cls_locs, roi_scores = self.head( # roi pooling + 全连接层网络，输入特征提取的feature map和rpn网络输出的rois,rois_indices
            h, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices

    def use_preset(self, preset): #预设超参数
        """Use the given preset during prediction.

        This method changes values of :obj:`self.nms_thresh` and
        :obj:`self.score_thresh`. These values are a threshold value
        used for non maximum suppression and a threshold value
        to discard low confidence proposals in :meth:`predict`,
        respectively.

        If the attributes need to be changed to something
        other than the values provided in the presets, please modify
        them by directly accessing the public attributes.

        Args:
            preset ({'visualize', 'evaluate'): A string to determine the
                preset to use.

        """
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def _suppress(self, raw_cls_bbox, raw_prob): #抑制输出(预测时使用)
        bbox = list() #最终的输出框
        label = list() #最终的输出label
        score = list() #最终的输出分数
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class): #忽略cls_id=0，因为是背景类。以类别为单位
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :] #该类别的bbox
            prob_l = raw_prob[:, l] #该类别概率
            mask = prob_l > self.score_thresh #第一轮筛选，得到分数大于阈值的索引
            cls_bbox_l = cls_bbox_l[mask] #得到需要的该类别的框
            prob_l = prob_l[mask] #得到需要的该类别概率
            keep = non_maximum_suppression( #第二轮筛选，非极大抑制，输入该类别的框和该类别概率
                cp.array(cls_bbox_l), self.nms_thresh, prob_l)
            keep = cp.asnumpy(keep) #得到需要的索引
            bbox.append(cls_bbox_l[keep]) #两轮筛选后的框
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))#label在[0,self.n_class - 2]为了对应索引,考虑背景和物理下标则减2
            score.append(prob_l[keep]) #两类筛选后的类别概率
        bbox = np.concatenate(bbox, axis=0).astype(np.float32) #最终的输出框
        label = np.concatenate(label, axis=0).astype(np.int32) #最终的输出label
        score = np.concatenate(score, axis=0).astype(np.float32) #最终的输出分数
        return bbox, label, score

    @nograd
    def predict(self, imgs,sizes=None,visualize=False): #预测函数
        """Detect objects from images.

        This method predicts objects for each image.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.

        Returns:
           tuple of lists:
           This method returns a tuple of three lists,
           :obj:`(bboxes, labels, scores)`.

           * **bboxes**: A list of float arrays of shape :math:`(R, 4)`, \
               where :math:`R` is the number of bounding boxes in a image. \
               Each bouding box is organized by \
               :math:`(y_{min}, x_{min}, y_{max}, x_{max})` \
               in the second axis.
           * **labels** : A list of integer arrays of shape :math:`(R,)`. \
               Each value indicates the class of the bounding box. \
               Values are in range :math:`[0, L - 1]`, where :math:`L` is the \
               number of the foreground classes.
           * **scores** : A list of float arrays of shape :math:`(R,)`. \
               Each value indicates how confident the prediction is.

        """
        self.eval() #网络设置为eval模式(禁用BatchNorm和Dropout)
        if visualize: #可视化内容，（跳过）
            self.use_preset('visualize')
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(at.tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
             prepared_imgs = imgs 
        bboxes = list() #最终的输出框
        labels = list() #最终的输出label
        scores = list() #最终的输出分数
        for img, size in zip(prepared_imgs, sizes): 
            img = at.totensor(img[None]).float() #增加batch维
            scale = img.shape[3] / size[1] #获得scale（待定）
            roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale) #前向
            # We are assuming that batch size is 1.
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            roi = at.totensor(rois) / scale #把rois变回原图尺寸（待定）

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = t.Tensor(self.loc_normalize_mean).cuda(). \
                repeat(self.n_class)[None]
            std = t.Tensor(self.loc_normalize_std).cuda(). \
                repeat(self.n_class)[None]
            #Q:看网上说ProposalCreator坐标归一化了所以这里要返回原图，但是我没看到。疑问
            #A:我觉得"ProposalCreator坐标归一化了"这个有错误，这里要反归一化是因为训练的时候使用的loc归一化了(ProposalTargetCreator)，所以预测结果loc是归一化后的，并不是ProposalCreator时候归一化了
            roi_cls_loc = (roi_cls_loc * std + mean) #坐标反归一化
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc) #一个框对应n_class个loc，所以要expand_as到同维度后面可以二次修正框
            
            #二次修正框得到最后框
            cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
                                at.tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = at.totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            # clip bounding box
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0]) #限制超出尺寸的框
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1]) #限制超出尺寸的框
            #softmax得到每个框的类别概率
            prob = at.tonumpy(F.softmax(at.totensor(roi_score), dim=1))

            raw_cls_bbox = at.tonumpy(cls_bbox)
            raw_prob = at.tonumpy(prob)
            #输入框以及对应的类别概率，抑制输出
            bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)
            
            #输出坐标，类别，该类别概率
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.use_preset('evaluate') #可视化内容，（跳过）
        self.train() #返回train模式
        return bboxes, labels, scores

    def get_optimizer(self): #得到optimizer函数
        """
        return optimizer, It could be overwriten if you want to specify 
        special optimizer
        """
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        if opt.use_adam:
            self.optimizer = t.optim.Adam(params)
        else:
            self.optimizer = t.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def scale_lr(self, decay=0.1): #衰减学习率函数
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer




