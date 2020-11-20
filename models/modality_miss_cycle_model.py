import torch
import os
import json
import random
import torch.nn.functional as F
from functools import reduce
from .base_model import BaseModel
from .early_fusion_multi_model import EarlyFusionMultiModel
from .networks.autoencoder import ResidualAE, ResidualUnetAE
from .networks.self_modules.fc_encoder import FcEncoder
from .networks.self_modules.lstm_encoder import LSTMEncoder
from .networks.self_modules.textcnn_encoder import TextCNN
from .networks.classifier import FcClassifier
from .networks.tools import init_net, MultiLayerFeatureExtractor
from .networks.self_modules.loss import MMD_loss, ContrastiveLoss, KLDivLoss_OnFeat
from .utils.config import OptConfig


class ModalityMissTestModel(BaseModel):
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
        parser.add_argument('--AE_layers', type=str, default='128,32,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--n_blocks', type=int, default=3, help='number of AE blocks')
        parser.add_argument('--output_dim', type=int, default=4, help='output dim')
        parser.add_argument('--dropout_rate', type=float, default=0.2, help='rate of dropout')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        parser.add_argument('--teacher_path', type=str, default='None')
        parser.add_argument('--ce_weight', type=float, default=1.0, help='weight of ce loss')
        parser.add_argument('--mmd_loss_type', type=str, default='KLdiv', choices=['MMD', 'KLdiv', 'MSE', 'CST'], help='mmd_loss type')
        parser.add_argument('--mmd_weight', type=float, default=1e-3, help='weight of mmd loss')
        parser.add_argument('--cycle_weight', type=float, default=1e-3, help='weight of mmd loss')
        parser.add_argument('--miss2_rate', type=float, default=0.2, help='# for miss_num=mix, probability of data which misses 2 modality')
        parser.add_argument('--real_data_rate', type=float, default=0.2, help='# of probability to use no missing data to train model')
        parser.add_argument('--niter_annealing', type=int, default=60, help='# until which epoch use missing modality logits')
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--modality', type=str, help='which modality to use for model')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        # define teacher network
        self.teacher_path = os.path.join(opt.teacher_path, str(opt.cvNo))
        self.teacher_config_path = os.path.join(opt.teacher_path, 'train_opt.conf')
        self.teacher_config = self.load_from_opt_record(self.teacher_config_path)
        self.teacher_config.isTrain = False                             # teacher model should be in test mode
        self.teacher_config.gpu_ids = opt.gpu_ids                       # set gpu to the same
        self.teacher_model = EarlyFusionMultiModel(self.teacher_config)
        self.teacher_model.cuda()
        self.teacher_model.load_networks_cv(self.teacher_path)
        self.teacher_model.eval()

        # define model settings 
        self.loss_names = ['CE', 'mmd', 'cycle']
        self.model_names = ['A', 'V', 'L', 'C', 'AE', 'AE_cycle', 'Miss']

        # define network autoencoder
        AE_layers = list(map(lambda x: int(x), opt.AE_layers.split(',')))
        self.netAE = ResidualAE(AE_layers, opt.n_blocks, opt.fusion_size, dropout=opt.dropout_rate, use_bn=opt.bn)
        self.netAE_cycle = ResidualAE(AE_layers, opt.n_blocks, opt.fusion_size, dropout=opt.dropout_rate, use_bn=opt.bn)
        
        # define network classifier
        fusion_layers = list(map(lambda x: int(x), opt.mid_layers_fusion.split(',')))
        self.netC = FcClassifier(opt.fusion_size, fusion_layers, opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        
        # define network Miss(maybe CNN?)
        input_size = AE_layers[-1] * opt.n_blocks
        self.netMiss = FcEncoder(input_size, [int((input_size + opt.fusion_size)/2), opt.fusion_size], dropout=opt.dropout_rate, use_bn=opt.bn)
        
        # acoustic model
        layers = list(map(lambda x: int(x), opt.mid_layers_a.split(',')))
        self.netA = FcEncoder(opt.input_dim_a, layers, opt.dropout_rate, opt.bn)
            
        # lexical model
        self.netL = TextCNN(opt.input_dim_l, opt.embd_size_l)
            
        # visual model
        self.netV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
            
        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            self.criterion_mmd = {
                'KLdiv': KLDivLoss_OnFeat(),
                'MMD': MMD_loss(),
                'MSE': torch.nn.MSELoss(),
                'CST': ContrastiveLoss()
            }[opt.mmd_loss_type]
            self.mmd_loss_type = opt.mmd_loss_type
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.ce_weight = opt.ce_weight
            self.mmd_weight = opt.mmd_weight
            self.cycle_weight = opt.cycle_weight
            self.output_dim = opt.output_dim
            self.real_data_rate = opt.real_data_rate
            self.miss_num = opt.miss_num
            self.miss2_rate = opt.miss2_rate
            self.cur_epoch = 1
            self.total_epoch = opt.niter + opt.niter_decay
            self.niter_annealing = opt.niter_annealing

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
        self.acoustic = input['acoustic'].float().to(self.device)
        self.lexical = input['lexical'].float().to(self.device)
        self.visual = input['visual'].float().to(self.device)
        self.label = input['label'].to(self.device)
        self.input = input
        
        if self.isTrain:
            batch_size = self.acoustic.size(0)
            self.miss_index = torch.zeros([batch_size]).long().to(self.device)
            self.miss_index_matrix = torch.zeros([batch_size, 3]).to(self.device)
            self.miss_index = self.miss_index.random_(0, 3)
            self.miss_index = self.miss_index_matrix.scatter_(1, self.miss_index.unsqueeze(1), 1).long()
    
            if self.miss_num == '1':
                self.miss_index = -1 * (self.miss_index - 1)  # 取反, 只抹去一个modality
        
            elif self.miss_num == 'mix':
                self.miss_index[:int((1-self.miss2_rate) * batch_size)] = -1 * \
                                (self.miss_index[:int((1-self.miss2_rate) * batch_size)] - 1) # 前(1-self.miss2_rate)的数据取反, 

            self.acoustic_miss = self.acoustic * self.miss_index[:, 0].unsqueeze(-1).float()
            self.lexical_miss = self.lexical * self.miss_index[:, 1].view([batch_size, 1, 1]).float()
            self.visual_miss = self.visual * self.miss_index[:, 2].view([batch_size, 1, 1]).float()
            # add real data
            real_index = random.sample(range(batch_size), int(batch_size * self.real_data_rate))
            self.acoustic_miss[real_index] = self.acoustic[real_index]
            self.lexical_miss[real_index] = self.lexical[real_index]
            self.visual_miss[real_index] = self.visual[real_index]
        else:
            self.acoustic_miss = self.acoustic
            self.visual_miss = self.visual
            self.lexical_miss = self.lexical
        
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # A modality
        self.feat_A_miss = self.netA(self.acoustic_miss)
        # L modality
        self.feat_L_miss = self.netL(self.lexical_miss)
        # V modality
        self.feat_V_miss = self.netV(self.visual_miss)
        # fusion miss
        self.feat_fusion_miss = torch.cat([self.feat_A_miss, self.feat_L_miss, self.feat_V_miss], dim=-1)
        # calc reconstruction of teacher's output
        self.recon_fusion, self.latent = self.netAE(self.feat_fusion_miss)
        # cycle consistency
        self.feat_cycle, self.latent_cycle = self.netAE_cycle(self.recon_fusion)
        # add up missing modality information
        self.miss_modality_feat = self.netMiss(self.latent)
        # get fusion outputs for missing modality
        self.logits, _ = self.netC(self.feat_fusion_miss + self.miss_modality_feat)
        self.pred = F.softmax(self.logits, dim=-1)

        # for training 
        if self.isTrain:
            with torch.no_grad():
                self.teacher_model.set_input(self.input)
                self.teacher_model.test()
                self.teacher_ef_fusion = self.teacher_model.feat

    def backward(self):
        """Calculate the loss for back propagation"""
        self.loss_CE = self.ce_weight * self.criterion_ce(self.logits, self.label)
        batch_size = self.label.size(0)
        if self.mmd_loss_type == 'CST':
            sample_num = 4
            self.loss_mmd = self.mmd_weight * self.criterion_mmd(self.recon_fusion, self.teacher_ef_fusion, torch.ones(batch_size).to(self.device))
            for random_idx in random.sample(range(1, batch_size-1), sample_num):
                re_rank_feat = torch.cat([self.teacher_ef_fusion[random_idx:], self.teacher_ef_fusion[:random_idx]])
                self.loss_mmd += self.mmd_weight * self.criterion_mmd(self.recon_fusion, re_rank_feat, torch.zeros(batch_size).to(self.device))
            
            self.loss_mmd /= (sample_num + 1)
        
        elif self.mmd_loss_type == 'MSE':
            self.loss_mmd = self.mmd_weight * self.criterion_mmd(self.recon_fusion, self.teacher_ef_fusion) 
        
        else:
            # positive pairs
            self.loss_mmd = self.mmd_weight * self.criterion_mmd(self.recon_fusion, self.teacher_ef_fusion) 
            # negative pairs
            sample_num = 4
            for random_idx in random.sample(range(1, batch_size-1), sample_num):
                re_rank_feat = torch.cat([self.teacher_ef_fusion[random_idx:], self.teacher_ef_fusion[:random_idx]])
                self.loss_mmd += self.mmd_weight * torch.clamp(1 - self.criterion_mmd(self.recon_fusion, re_rank_feat), min=0)            

            self.loss_mmd /= (sample_num + 1)
        
        self.loss_cycle = self.cycle_weight * self.criterion_mmd(self.feat_cycle, self.feat_fusion_miss)

        loss = self.loss_CE + self.loss_mmd + self.loss_cycle
        loss.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 0.1)

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()   
        # backward
        self.optimizer.zero_grad()  
        self.backward()            
        self.optimizer.step()
        # update cur_epoch:
        self.cur_epoch = epoch
    
    def load_from_opt_record(self, file_path):
        opt_content = json.load(open(file_path, 'r'))
        opt = OptConfig()
        opt.load(opt_content)
        return opt
            
