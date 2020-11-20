import torch
import os
import random
import json
import copy
import torch.nn.functional as F
from collections import OrderedDict
from .base_model import BaseModel
from .networks.self_modules.fc_encoder import FcEncoder
from .networks.self_modules.lstm_encoder import LSTMEncoder
from .early_fusion_multi_model import EarlyFusionMultiModel
from .multi_fusion_multi_model import MultiFusionMultiModel
from .networks.self_modules.textcnn_encoder import TextCNN
from .networks.autoencoder import ResidualAE, SimpleFcAE
from .networks.classifier import FcClassifier
from .networks.tools import init_net
from .utils.config import OptConfig

class NewSimpleAEModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # parser.set_defaults(no_dropout=True)
        parser.add_argument('--input_dim_a', type=int, default=1582, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=384, help='lexical input dim')
        parser.add_argument('--mid_layers_a', type=str, default='256,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--hidden_size_v', default=128, type=int, help='rnn hidden layer')
        parser.add_argument('--embd_size_v', default=128, type=int, help='embedding size for v')
        parser.add_argument('--embd_size_l', default=128, type=int, help='embedding size for l')
        parser.add_argument('--embd_method', default='maxpool', type=str, help='LSTM encoder embd function')
        parser.add_argument('--fusion_size', type=int, default=384, help='fusion model fusion size')
        parser.add_argument('--mid_layers_fusion', type=str, default='128,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--AE_layers', type=str, default='128,64,32', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--n_blocks', type=int, default=3, help='number of AE blocks')
        parser.add_argument('--ce_weight', type=float, default=1.0, help='weight of ce loss')
        parser.add_argument('--mse_weight', type=float, default=1.0, help='weight of mse loss')
        parser.add_argument('--cycle_weight', type=float, default=1.0, help='weight of cycle loss')
        parser.add_argument('--teacher_path', type=str, default='None')
        parser.add_argument('--output_dim', type=int, default=4, help='output dim')
        parser.add_argument('--dropout_rate', type=float, default=0.2, help='rate of dropout')
        parser.add_argument('--miss2_rate', type=float, default=0.2, help='probability of data which misses 2 modality')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in DNN')
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--modality', type=str, help='which modality to use [AVL]')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        self.loss_names = ['CE', 'mse', 'cycle']
        self.modality = opt.modality
        self.model_names = ['A', 'V', 'L', 'C', 'AE', 'AE_cycle']
        
        # acoustic model
        A_layers = list(map(lambda x: int(x), opt.mid_layers_a.split(',')))
        self.netA = FcEncoder(opt.input_dim_a, A_layers, opt.dropout_rate, opt.bn)
        # lexical model
        self.netL = TextCNN(opt.input_dim_l, opt.embd_size_l)
        # visual model
        self.netV = LSTMEncoder(opt.input_dim_v, opt.hidden_size_v, opt.embd_method)
        # AE model
        AE_layers = list(map(lambda x: int(x), opt.AE_layers.split(',')))
        input_dim = A_layers[-1] + opt.hidden_size_v + opt.embd_size_l
        self.netAE = SimpleFcAE(input_dim, AE_layers, dropout=0, use_bn=False)
        self.netAE_cycle = SimpleFcAE(input_dim, AE_layers, dropout=0, use_bn=False)
        # classifier
        fusion_layers = list(map(lambda x: int(x), opt.mid_layers_fusion.split(',')))
        self.netC = FcClassifier(AE_layers[-1] * opt.n_blocks, fusion_layers, opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        # init parameter from pretrained model
        self.init_from_exist_model(opt)

        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            self.criterion_mse = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.ce_weight = opt.ce_weight
            self.mse_weight = opt.mse_weight
            self.cycle_weight = opt.cycle_weight
            self.miss2_rate = opt.miss2_rate
        
        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        
    def init_from_exist_model(self, opt):
        print('Init parameter from {}'.format(opt.teacher_path))
        teacher_path = os.path.join(opt.teacher_path, str(opt.cvNo))
        teacher_config_path = os.path.join(opt.teacher_path, 'train_opt.conf')
        teacher_config = self.load_from_opt_record(teacher_config_path)
        teacher_config.isTrain = False                             # teacher model should be in test mode
        teacher_config.gpu_ids = opt.gpu_ids                       # set gpu to the same
        self.teacher_model = EarlyFusionMultiModel(teacher_config)
        # self.teacher_model = MultiFusionMultiModel(teacher_config)
        self.teacher_model.load_networks_cv(teacher_path)
        self.teacher_model.cuda()
        self.teacher_model.eval()
        
    def post_process(self):
        # called after model.setup()
        def transform_key_for_parallel(state_dict):
            return OrderedDict([('module.'+key, value) for key, value in state_dict.items()])
        
        print('Load parameters from teacher network')
        f = lambda x: transform_key_for_parallel(x)
        self.netA.load_state_dict(f(self.teacher_model.netA.state_dict()))
        self.netV.load_state_dict(f(self.teacher_model.netV.state_dict()))
        self.netL.load_state_dict(f(self.teacher_model.netL.state_dict()))
    
    def load_from_opt_record(self, file_path):
        opt_content = json.load(open(file_path, 'r'))
        opt = OptConfig()
        opt.load(opt_content)
        return opt

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
            self.miss_index[:int((1-self.miss2_rate) * batch_size)] = -1 * \
                            (self.miss_index[:int((1-self.miss2_rate) * batch_size)] - 1) # 前(1-self.miss2_rate)的数据取反, 

            self.acoustic_miss = self.acoustic * self.miss_index[:, 0].unsqueeze(-1).float()
            self.lexical_miss = self.lexical * self.miss_index[:, 1].view([batch_size, 1, 1]).float()
            self.visual_miss = self.visual * self.miss_index[:, 2].view([batch_size, 1, 1]).float()
            # reverse data
            self.miss_index_reverse = -1 * (self.miss_index - 1)
            self.acoustic_reverse = self.acoustic * self.miss_index_reverse[:, 0].unsqueeze(-1).float()
            self.lexical_reverse = self.lexical * self.miss_index_reverse[:, 1].view([batch_size, 1, 1]).float()
            self.visual_reverse = self.visual * self.miss_index_reverse[:, 2].view([batch_size, 1, 1]).float()
            
            '''check'''
            # for i, (a, l, v, ar, lr, vr, mi, mr, aa, ll, vv) in enumerate(zip(self.acoustic_miss, self.lexical_miss, self.visual_miss, \
            #     self.acoustic_reverse, self.lexical_reverse, self.visual_reverse, 
            #     self.miss_index, self.miss_index_reverse,
            #     self.acoustic, self.lexical, self.visual)):
            
            #     print('\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(i, 
            #         torch.sum(aa).long().item(), torch.sum(ll).long().item(), torch.sum(vv).long().item(),
            #         torch.sum(a).long().item(), torch.sum(l).long().item(), torch.sum(v).long().item(), 
            #         torch.sum(ar).long().item(), torch.sum(lr).long().item(), torch.sum(vr).long().item(), 
            #         mi.cpu().numpy(), mr.cpu().numpy()))
            
            # input()
            
        else:
            # in val and tst data is in missing type
            self.acoustic_miss = self.acoustic_reverse = self.acoustic
            self.visual_miss = self.visual_reverse = self.visual
            self.lexical_miss = self.lexical_reverse = self.lexical
         
    
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
        self.recon_cycle, self.latent_cycle = self.netAE_cycle(self.recon_fusion)
        # get fusion outputs for missing modality
        self.logits, _ = self.netC(self.latent)
        self.pred = F.softmax(self.logits, dim=-1)
        # for training 
        if self.isTrain:
            with torch.no_grad():
                self.T_embd_A = self.teacher_model.netA(self.acoustic_reverse)
                self.T_embd_L = self.teacher_model.netL(self.lexical_reverse)
                self.T_embd_V = self.teacher_model.netV(self.visual_reverse)
                self.T_embds = torch.cat([self.T_embd_A, self.T_embd_L, self.T_embd_V], dim=-1)

    def backward(self):
        """Calculate the loss for back propagation"""
        # calc CE_loss 
        self.loss_CE = self.ce_weight * self.criterion_ce(self.logits, self.label)
        self.loss_mse = self.mse_weight * self.criterion_mse(self.T_embds, self.recon_fusion)
        self.loss_cycle = self.cycle_weight * self.criterion_mse(self.feat_fusion_miss.detach(), self.recon_cycle)
        loss = self.loss_CE + self.loss_mse + self.loss_cycle
        loss.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 5)

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