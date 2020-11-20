import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)

        bandwidth /= kernel_mul ** (kernel_num // 2) 
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / (bandwidth_temp + 1e-6) ) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        # print("FIRST",source[0][:5], target[0][:5])
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)

        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        
        return loss

class KLDivLoss_OnFeat(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.KLDivLoss()
    
    def forward(self, feat1, feat2):
        # feat1 = F.log_softmax(feat1, dim=-1)
        # feat2 = F.softmax(feat2, dim=-1)
        return self.loss(feat1, feat2)

class SoftCenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Change it to a soft label version.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=4, feat_dim=128, use_gpu=True):
        super(SoftCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        init_tensor = torch.Tensor(self.num_classes, self.feat_dim)
        torch.nn.init.xavier_normal_(init_tensor)
        # whether use tensor.cuda()
        if self.use_gpu:
            self.centers = nn.Parameter(init_tensor).cuda()
        else:
            self.centers = nn.Parameter(init_tensor)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: soft label with shape (batch_size, class_num).
        """
        batch_size, feat_dim = x.size()
        # x expand to size [class_num, batch_size, feat_dim]
        x_mat = x.expand(self.num_classes, batch_size, feat_dim)
        # center expand to size [class_num, batch_size, feat_dim]
        center_mat = self.centers.unsqueeze(1).expand(self.num_classes, batch_size, feat_dim)
        # calculate the square error using "torch.sum((x_mat - center_mat)**2, dim=-1)"
        # and then do the weighted sum using weight labels
        delta = torch.sum((x_mat - center_mat)**2, dim=-1) * labels.t()
        loss = torch.sum(delta) / batch_size
        return loss


class SpectralLoss(nn.Module):
    ''' Calculate spectral loss 
        L_{spec} = mean(wij * ||Yi - Yj||_2) for each pair in a mini-batch.
    '''
    def __init__(self, adjacent):
        super().__init__()
        self.adjacent = torch.from_numpy(adjacent).cuda().float()
        self.epsilon = 1e-6

    def forward(self, batch_data, batch_indexs):
        ''' batch_data: [batch_size, feat_dim]
        '''
        batch_size = batch_data.size(0)
        feat_dim = batch_data.size(1)
        ai = batch_data.expand(batch_size, batch_size, feat_dim)
        aj = ai.transpose(0, 1)
        local_adjacent = self.adjacent[batch_indexs][:, batch_indexs]
        loss = torch.sum(torch.sqrt(torch.sum((ai-aj)**2, dim=2) + self.epsilon) * local_adjacent)
        return loss / (batch_size * batch_size)

        # batch_size = batch_data.size(0)
        # feat_dim = batch_data.size(1)
        # local_adjacent = self.adjacent[batch_indexs][:, batch_indexs]
        # total_loss = torch.as_tensor(0.0).cuda()
        # for i in range(batch_size):
        #     for j in range(batch_size):
        #         weight = local_adjacent[i, j]
        #         total_loss += weight * torch.dist(batch_data[i], batch_data[j], p=2)
        # return total_loss / (batch_size * batch_size)

class OrthPenalty(nn.Module):
    ''' Calculate orth penalty
        if input batch of feat is Y with size [batch_size, feat_dim]
        L_{orth} = sum(|Y@Y.T - I|) / batch_size**2 
                   where I is a diagonal matrix with size [batch_size, batch_size]
    '''
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-6
    
    def forward(self, batch_data):
        ''' batch_data: [batch_size, feat_dim]
        '''
        batch_size = batch_data.size(0)
        feat_dim = batch_data.size(1)
        I = torch.eye(feat_dim).cuda() * batch_size
        loss = torch.sum(torch.sqrt(((batch_data.transpose(0, 1) @ batch_data) - I + self.epsilon)**2))
        return loss / (batch_size) 
       

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y):
        self.check_type_forward((x0, x1, y))

        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq + 1e-6)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss
    

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None