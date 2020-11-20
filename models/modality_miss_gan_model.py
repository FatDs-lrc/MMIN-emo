import torch
import os
import json
import random
import torch.nn.functional as F
from .base_model import BaseModel
from .early_fusion_multi_model import EarlyFusionMultiModel
from .networks.self_modules.fc_encoder import FcEncoder
from .networks.self_modules.lstm_encoder import LSTMEncoder
from .networks.self_modules.textcnn_encoder import TextCNN
from .networks.classifier import FcClassifier
from .networks.tools import init_net, MultiLayerFeatureExtractor
from .networks.self_modules.loss import MMD_loss, SoftCenterLoss, KLDivLoss_OnFeat
from .utils.config import OptConfig

'''
GAN 来实现模态缺失的模型构建, 使输入缺失模态的时候的输出靠近完整模态的输入
'''
class ModalityMissTSModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--mid_layers_a', type=str, default='256,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--input_dim_a', type=int, default=1582, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=384, help='lexical input dim')
        parser.add_argument('--embd_size_a', default=128, type=int, help='audio model embedding size')
        parser.add_argument('--embd_size_l', default=128, type=int, help='text model embedding size')
        parser.add_argument('--embd_size_v', default=128, type=int, help='visual model embedding size')
        parser.add_argument('--hidden_size_v', default=128, type=int, help='rnn hidden layer')
        parser.add_argument('--embd_method_v', default='last', type=str, help='visual embedding method,last,mean or atten')
        parser.add_argument('--fusion_size', type=int, default=384, help='fusion model fusion size')
        parser.add_argument('--mid_layers_fusion', type=str, default='128,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--output_dim', type=int, default=4, help='output dim')
        parser.add_argument('--dropout_rate', type=float, default=0.2, help='rate of dropout')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        parser.add_argument('--teacher_path', type=str, default='None')
        parser.add_argument('--lambda_gp', type=float, default=10.0, help='weight of gradient penalty')
        parser.add_argument('--n_critic', type=int, default=5, help='number of critic iterations per generator iteration')
        parser.add_argument('--ce_weight', type=float, default=1.0, help='weight of ce loss')
        parser.add_argument('--miss2_rate', type=float, default=0.2, help='# for miss_num=mix, probability of data which misses 2 modality')
        parser.add_argument('--real_data_rate', type=float, default=0.2, help='# of probability to use no missing data to train student model')
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--modality', type=str, help='which modality to use for student model')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        self.teacher_path = os.path.join(opt.teacher_path, str(opt.cvNo))
        self.teacher_config_path = os.path.join(opt.teacher_path, 'train_opt.conf')
        self.teacher_config = self.load_from_opt_record(self.teacher_config_path)
        self.teacher_config.isTrain = False                             # teacher model should be in test mode
        self.teacher_config.gpu_ids = opt.gpu_ids                       # set gpu to the same
        self.teacher_model = EarlyFusionMultiModel(self.teacher_config)
        self.teacher_model.cuda()
        self.teacher_model.load_networks_cv(self.teacher_path)
        self.loss_names = ['loss_G', 'loss_D', 'loss_gp']
        self.modality = opt.modality
        self.model_names = ['C', 'D']
        fusion_layers = list(map(lambda x: int(x), opt.mid_layers_fusion.split(',')))
        self.netC = FcClassifier(opt.fusion_size, fusion_layers, opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        self.netD = FcClassifier(opt.fusion_size, fusion_layers, output_dim=1, dropout=opt.dropout_rate, use_bn=False)
        # acoustic model
        if 'A' in self.modality:
            self.model_names.append('A')
            layers = list(map(lambda x: int(x), opt.mid_layers_a.split(',')))
            self.netA = FcEncoder(opt.input_dim_a, layers, opt.dropout_rate, opt.bn)
            
        # lexical model
        if 'L' in self.modality:
            self.model_names.append('L')
            self.netL = TextCNN(opt.input_dim_l, opt.embd_size_l)
            
        # visual model
        if 'V' in self.modality:
            self.model_names.append('V')
            self.netV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
            
        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()
    
            self.criterion_mmd = {
                'KLdiv': KLDivLoss_OnFeat(),
                'MMD': MMD_loss(),
                'MSE': torch.nn.MSELoss(),
            }[opt.mmd_loss_type]
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters_G = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names if net != 'D']
            self.optimizer_G = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers += [self.optimizer_G, self.optimizer_D]
            self.ce_weight = opt.ce_weight
            self.mmd_weight = opt.mmd_weight
            self.teacher_mmd_layers = opt.teacher_mmd_layers
            self.student_mmd_layers = opt.student_mmd_layers
            self.cur_critic = 0                                 # current critic iteration
            self.output_dim = opt.output_dim
            self.real_data_rate = opt.real_data_rate
            self.lambda_gp = opt.lambda_gp
            self.n_critic = opt.n_critic
            if self.ce_weight > 0:
                self.loss_names.append('CE')
    
            if self.mmd_weight > 0:
                self.loss_names.append('mmd')
                self.teacher_extractor = MultiLayerFeatureExtractor(self.teacher_model, self.teacher_mmd_layers)
                self.student_extractor = MultiLayerFeatureExtractor(self, self.student_mmd_layers)

            self.miss_num = opt.miss_num
            self.miss2_rate = opt.miss2_rate

        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
    
    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        # for teacher, raw data no modality missing
        self.acoustic = input['acoustic'].float().cuda()
        self.lexical = input['lexical'].float().cuda()
        self.visual = input['visual'].float().cuda()
        self.label = input['label'].cuda()
        self.input = input
        
        if self.isTrain and random.random() > self.real_data_rate:
            batch_size = self.acoustic.size(0)
            self.miss_index = torch.zeros([batch_size]).long()
            self.miss_index_matrix = torch.zeros([batch_size, 3])
            if len(self.gpu_ids) > 0:
                self.miss_index = self.miss_index.cuda()
                self.miss_index_matrix = self.miss_index_matrix.cuda()
            
            self.miss_index = self.miss_index.random_(0, 3)
            self.miss_index = self.miss_index_matrix.scatter_(1, self.miss_index.unsqueeze(1), 1).long()
    
            if self.miss_num == '1':
                self.miss_index = -1 * (self.miss_index - 1)  # 取反, 只抹去一个modality
        
            elif self.miss_num == 'mix':
                self.miss_index[:int((1-self.miss2_rate) * batch_size)] = -1 * \
                                (self.miss_index[:int((1-self.miss2_rate) * batch_size)] - 1) # 前(1-self.miss2_rate)的数据取反, 

            self.acoustic_miss = self.acoustic * self.miss_index[:, 0].unsqueeze(-1).float()
            self.visual_miss = self.visual * self.miss_index[:, 1].view([batch_size, 1, 1]).float()
            self.lexical_miss = self.lexical * self.miss_index[:, 2].view([batch_size, 1, 1]).float()
        else:
            self.acoustic_miss = self.acoustic
            self.visual_miss = self.visual
            self.lexical_miss = self.lexical
            
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isTrain:
            with torch.no_grad():
                self.teacher_model.set_input(self.input)
                self.teacher_model.test()
                self.teacher_mid_feats = self.teacher_extractor.extract()
    
        final_embd = []
        if 'A' in self.modality:
            # self.student_logits, self.student_feat = self.netA(self.acoustic)
            self.student_feat_A = self.netA(self.acoustic_miss)
            final_embd.append(self.student_feat_A)

        if 'L' in self.modality:
            self.student_feat_L = self.netL(self.lexical_miss)
            final_embd.append(self.student_feat_L)
        
        if 'V' in self.modality:
            self.student_feat_V = self.netV(self.visual_miss)
            final_embd.append(self.student_feat_V)
        
        # get student outputs
        self.student_feat = torch.cat(final_embd, dim=-1)
        self.student_logits, _ = self.netC(self.student_feat)

        if self.isTrain:
            self.student_mid_feats = self.student_extractor.extract()

        else:
            self.pred = F.softmax(self.student_logits, dim=-1)
   
    def backward_G(self):
        """Calculate the loss for back propagation"""
        loss = torch.as_tensor(0.0)
        if len(self.gpu_ids) >0:
            loss = loss.cuda()

        if self.ce_weight > 0:
            self.loss_CE = self.ce_weight * self.criterion_ce(self.student_logits, self.label)
            loss += self.loss_CE

        if self.mmd_weight > 0:
            self.loss_mmd = torch.tensor([self.criterion_mmd(teacher_feat, student_feat) 
                            for (teacher_feat, student_feat) in zip(self.teacher_mid_feats, self.student_mid_feats)])
            self.loss_mmd = torch.sum(self.loss_mmd)
            loss += self.mmd_weight
        
        self.loss_G = self.netD(self.student_mid_feats).mean()
        loss += self.loss_G
        loss.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 0.1)
    
    def backward_D(self):
        real = self.teacher_mid_feats
        fake = self.student_mid_feats.detach()
        self.loss_d_real = self.netD(real).mean()
        self.loss_d_fake = self.netD(fake).mean()
        self.loss_gp = self.lambda_gp * GradientPenaltyLoss(self.netD, real, fake)
        self.loss_D = self.loss_d_fake - self.loss_d_real
        loss = self.loss_D + self.loss_gp
        loss.backward()

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward()   
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()    
        self.backward_D()        
        self.optimizer_D.step() 
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def load_from_opt_record(self, file_path):
        opt_content = json.load(open(file_path, 'r'))
        opt = OptConfig()
        opt.load(opt_content)
        return opt
