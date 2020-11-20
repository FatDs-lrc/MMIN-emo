import torch
import os
import torch.nn.functional as F
from .base_model import BaseModel
from .networks.self_modules.fc_encoder import FcEncoder
from .networks.self_modules.lstm_encoder import LSTMEncoder
from .networks.self_modules.textcnn_encoder import TextCNN
from .networks.classifier import FcClassifier
from .networks.tools import init_net

class CycleMappingModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # parser.set_defaults(no_dropout=True)
        parser.add_argument('--input_dim_a', type=int, default=1582, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--mid_layers_a', type=str, default='256,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--hidden_size', default=128, type=int, help='rnn hidden layer')
        parser.add_argument('--embd_size_l', default=128, type=int, help='embedding size for l')
        parser.add_argument('--embd_method', default='maxpool', type=str, help='LSTM encoder embd function')
        parser.add_argument('--fusion_size', type=int, default=384, help='fusion model fusion size')
        parser.add_argument('--mid_layers_fusion', type=str, default='128,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--mapping_layers', type=str, default='128,128,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--output_dim', type=int, default=4, help='output dim')
        parser.add_argument('--dropout_rate', type=float, default=0.2, help='rate of dropout')
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
        self.loss_names = ['CE_real', 'CE_fake', 'map']#, 'map', 'MSE_A2L', 'MSE_L2A',]
        self.model_names = ['A', 'L', 'A_C', 'L_C', 'C', 'A2L', 'L2A']
        # fusion classifier
        fusion_layers = list(map(lambda x: int(x), opt.mid_layers_fusion.split(',')))
        self.netC = FcClassifier(opt.fusion_size, fusion_layers, opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        # acoustic model
        A_layers = list(map(lambda x: int(x), opt.mid_layers_a.split(',')))
        self.netA = FcEncoder(opt.input_dim_a, A_layers, opt.dropout_rate, opt.bn)
        self.netA_C = FcClassifier(A_layers[-1], fusion_layers, opt.output_dim, dropout=opt.dropout_rate)
        # lexical model
        self.netL = TextCNN(opt.input_dim_l, opt.embd_size_l)
        self.netL_C = FcClassifier(opt.embd_size_l, fusion_layers, opt.output_dim, dropout=opt.dropout_rate)
        # net A2L
        mapping_layers = list(map(lambda x: int(x), opt.mapping_layers.split(',')))
        self.netA2L = FcEncoder(A_layers[-1], mapping_layers, opt.dropout_rate, opt.bn)
        # net L2A
        self.netL2A = FcEncoder(opt.embd_size_l, mapping_layers, opt.dropout_rate, opt.bn)
        # define training settings 
        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            self.criterion_mse = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters_cls = [{'params': getattr(self, 'net'+net).parameters()} for net in ['A', 'L', 'A_C', 'L_C', 'C']]
            self.optimizer_cls = torch.optim.Adam(paremeters_cls, lr=opt.lr, betas=(opt.beta1, 0.999))
            parameters_map = [{'params': getattr(self, 'net'+net).parameters()} for net in ['A2L', 'L2A']]
            self.optimizer_map = torch.optim.Adam(parameters_map, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer_cls, self.optimizer_map]
            # define flag to control whether use missing data during training
            self.training_miss = False
    
        self.output_dim = opt.output_dim
        self.fusion_embd_size = fusion_layers[-1]
        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
    
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.acoustic = input['acoustic'].float().cuda()
        self.lexical = input['lexical'].float().cuda()
        self.label = input['label'].cuda()
        if not self.isTrain:
            self.miss_index = input['miss_index']
        
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isTrain:
            # for A input
            self.embd_A = self.netA(self.acoustic)
            self.logits_A, self.feat_A = self.netA_C(self.embd_A)
            self.pred_A = F.softmax(self.logits_A, dim=-1)
            # for L input
            self.embd_L = self.netL(self.lexical)
            self.logits_L, self.feat_L = self.netL_C(self.embd_L)
            self.pred_L = F.softmax(self.logits_L, dim=-1)
            # mapping using A data
            self.LfromA = self.netA2L(self.embd_A)
            self.logits_LfromA, self.feat_LfromA = self.netL_C(self.LfromA)
            self.A_cycle = self.netL2A(self.LfromA)
            # mapping using L data
            self.AfromL = self.netL2A(self.embd_L)
            self.logits_AfromL, self.feat_AfromL = self.netA_C(self.AfromL)
            self.L_cycle = self.netA2L(self.AfromL)
            # fusion (early fusion && late fusions)
            # [A, L]
            self.ef_reps = torch.cat([self.embd_A, self.embd_L], dim=-1)
            self.ef_logits, self.ef_fusion_feat = self.netC(self.ef_reps)
            self.ef_preds = F.softmax(self.ef_logits, dim=-1)
            # [A, LfromA]
            self.ef_reps_A_RL = torch.cat([self.embd_A, self.LfromA], dim=-1)
            self.ef_logits_A_RL, self.ef_fusion_feat_A_RL = self.netC(self.ef_reps_A_RL)  
            self.ef_preds_A_RL = F.softmax(self.ef_logits_A_RL, dim=-1)
            # [AfromL, L]
            self.ef_reps_RA_L = torch.cat([self.AfromL, self.embd_L], dim=-1)
            self.ef_logits_RA_L, self.ef_fusion_feat_RA_L = self.netC(self.ef_reps_RA_L) 
            self.ef_preds_RA_L = F.softmax(self.ef_logits_RA_L, dim=-1) 
            # [AfromL, LfromA]
        else:
            # total logits and preds
            batch_size = self.label.size(0)
            self.ef_logits = torch.Tensor(batch_size, self.output_dim).to(self.device)
            self.ef_preds = torch.Tensor(batch_size, self.output_dim).to(self.device)
            self.ef_fusion_feat = torch.Tensor(batch_size, self.fusion_embd_size).to(self.device)
            '''for input which only have A data'''
            self.A_index = self.miss_index[:, 0].nonzero().squeeze()
            if len(self.A_index) > 0:
                # calc A embd etc.
                self.embd_A = self.netA(self.acoustic[self.A_index])
                self.logits_A, self.feat_A = self.netA_C(self.embd_A)
                self.pred_A = F.softmax(self.logits_A, dim=-1)
                # calc mapping A->L
                self.LfromA = self.netA2L(self.embd_A)
                self.A_cycle = self.netL2A(self.LfromA)
                # fusion
                self.ef_reps_fromA = torch.cat([self.embd_A, self.LfromA], dim=-1)
                self.ef_logits_fromA, self.ef_fusion_feat_fromA = self.netC(self.ef_reps_fromA)
                self.ef_preds_fromA = F.softmax(self.ef_logits_fromA, dim=-1)
                # fill into total logits, preds, feats
                self.ef_logits[self.A_index] = self.ef_logits_fromA
                self.ef_preds[self.A_index] = self.ef_preds_fromA
                self.ef_fusion_feat[self.A_index] = self.ef_fusion_feat_fromA

            ''' for input which only have L data '''
            self.L_index = self.miss_index[:, 1].nonzero().squeeze()
            if len(self.L_index) > 0:
                # calc L embd etc.
                self.embd_L = self.netL(self.lexical[self.L_index])
                self.logits_L, self.feat_L = self.netL_C(self.embd_L)
                self.pred_L = F.softmax(self.logits_L, dim=-1)
                # calc mapping L->A
                self.AfromL = self.netL2A(self.embd_L)
                self.L_cycle = self.netA2L(self.AfromL)
                # fusion
                self.ef_reps_fromL = torch.cat([self.AfromL, self.embd_L], dim=-1)
                self.ef_logits_fromL, self.ef_fusion_feat_fromL = self.netC(self.ef_reps_fromL)
                self.ef_preds_fromL = F.softmax(self.ef_logits_fromL, dim=-1)
                # fill into total logits, preds, feats
                self.ef_logits[self.L_index] = self.ef_logits_fromL
                self.ef_preds[self.L_index] = self.ef_preds_fromL
                self.ef_fusion_feat[self.L_index] = self.ef_fusion_feat_fromL
            ''' end of forward in test phase'''

        # only use early fusion pred
        self.pred = self.ef_preds
        # late fusion of preds
        # self.pred = torch.stack(final_pred)
        # self.pred = torch.mean(self.pred, dim=0)

    def backward_real_cls(self):
        """Calculate the classification loss for real data input"""
        self.loss_CE_F = self.criterion_ce(self.ef_logits, self.label)
        self.loss_CE_A = self.criterion_ce(self.logits_A, self.label)
        self.loss_CE_L = self.criterion_ce(self.logits_L, self.label)
        self.loss_CE_real = self.loss_CE_F + self.loss_CE_A + self.loss_CE_L
        self.loss_CE_real.backward(retain_graph=True)
        for model in ['A', 'L', 'C', 'A_C', 'L_C']:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 5)

    def backward_mapping(self):
        """Calculate the loss for modality mapping"""
        self.loss_MSE_A2L = self.criterion_mse(self.embd_L, self.LfromA)
        # self.loss_cycle_A = self.criterion_mse(self.embd_A, self.A_cycle)
        self.loss_MSE_L2A = self.criterion_mse(self.embd_A, self.AfromL)
        # self.loss_cycle_L = self.criterion_mse(self.embd_L, self.L_cycle)
        self.loss_map = self.loss_MSE_A2L + self.loss_MSE_L2A
        self.loss_map.backward(retain_graph=False)
        for model in ['A2L', 'L2A']:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 5)
    
    def backward_fake_cls(self):
        self.loss_CE_F_A_RL = self.criterion_ce(self.ef_logits_A_RL, self.label)
        self.loss_CE_F_RA_L = self.criterion_ce(self.ef_logits_RA_L, self.label)
        self.loss_CE_AfromL = self.criterion_ce(self.logits_AfromL, self.label)
        self.loss_CE_LfromA = self.criterion_ce(self.logits_LfromA, self.label)
        self.loss_CE_fake = self.loss_CE_F_A_RL + self.loss_CE_F_RA_L + self.loss_CE_AfromL + self.loss_CE_LfromA
        self.loss_CE_fake.backward(retain_graph=True)
        for model in ['C', 'A_C', 'L_C', 'A2L', 'L2A']:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 5)

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()   
        # backward netA and netL
        # self.set_requires_grad([self.netA, self.netL], True)
        self.optimizer_cls.zero_grad()  
        self.backward_real_cls()            
        self.backward_fake_cls() 
        self.optimizer_cls.step()
        # backward netA2L netL2A
        self.optimizer_map.zero_grad()
        self.set_requires_grad([self.netA, self.netL], False)
        self.backward_mapping()
        self.optimizer_map.step()
