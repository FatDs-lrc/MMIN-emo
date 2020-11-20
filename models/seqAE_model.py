import torch
import os
import random
import json
import copy
import torch.nn.functional as F
from .base_model import BaseModel
from .networks.self_modules.fc_encoder import FcEncoder
from .networks.self_modules.lstm_encoder import LSTMEncoder, LSTMSeqDecoder
from .multi_fusion_multi_model import MultiFusionMultiModel
from .networks.self_modules.textcnn_encoder import TextCNN
from .networks.autoencoder import ResidualAE, SimpleFcAE
from .networks.classifier import FcClassifier
from .networks.tools import init_net
from .utils.config import OptConfig

class SeqAEModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # parser.set_defaults(no_dropout=True)
        parser.add_argument('--input_modality', type=str, default="A")
        parser.add_argument('--output_modality', type=str, default="L")
        parser.add_argument('--A_layers', type=str, default='512,256')
        parser.add_argument('--embd_size', default=128, type=int, help='embedding size')
        parser.add_argument('--embd_method', default='maxpool', type=str, help='LSTM encoder embd function')
        parser.add_argument('--fusion_size', type=int, default=384, help='fusion model fusion size')
        parser.add_argument('--mid_layers_fusion', type=str, default='128,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--ce_weight', type=float, default=1.0, help='weight of ce loss')
        parser.add_argument('--mse_weight', type=float, default=1.0, help='weight of mse loss')
        parser.add_argument('--cycle_weight', type=float, default=1.0, help='weight of cycle loss')
        parser.add_argument('--output_dim', type=int, default=4, help='output dim')
        parser.add_argument('--dropout_rate', type=float, default=0.2, help='rate of dropout')
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        # self.teacher_path = os.path.join(opt.teacher_path, f'cvNo{opt.cvNo}')
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        # self.loss_names = ['CE']
        self.loss_names = ['CE', 'mse']
        self.model_names = ['enc', 'dec', 'C']
        fusion_layers = list(map(lambda x: int(x), opt.mid_layers_fusion.split(',')))
        self.netC = FcClassifier(opt.embd_size, fusion_layers, opt.output_dim, dropout=opt.dropout_rate, use_bn=False)
        self.input_modality = opt.input_modality
        self.output_modality = opt.output_modality
        assert self.input_modality != self.output_modality
        self.modality_dim = {'A': 1582, 'V': 342, 'L': 1024}
        self.seq_len = {'V': 50, 'L': 22}
        # net C
        self.netC = FcClassifier(opt.embd_size, [128, 128], opt.output_dim, dropout=opt.dropout_rate)
        # encoder model
        if opt.input_modality == 'A':
            A_layers = list(map(lambda x: int(x), opt.A_layers.split(','))) + [opt.embd_size]
            self.netenc = FcEncoder(1582, A_layers, dropout=opt.dropout_rate)
        else:
            self.netenc = LSTMEncoder(self.modality_dim[self.input_modality], opt.embd_size, embd_method=opt.embd_method)
        
        # decoder model
        if opt.output_modality == 'A':
            A_layers = list(map(lambda x: int(x), opt.A_layers.split(','))) + [self.modality_dim[self.output_modality]]
            self.netdec = FcEncoder(opt.embd_size, A_layers, dropout=opt.dropout_rate)
        else:
            self.netdec = LSTMSeqDecoder(opt.embd_size, self.modality_dim[self.output_modality], 
                                        self.seq_len[self.output_modality])

        # # cycle encoder
        # if opt.input_modality == 'A':
        #     A_layers = list(map(lambda x: int(x), opt.A_layers.split(','))) + opt.embd_size
        #     self.cycle_encoder = FcEncoder(1582, A_layers, dropout=opt.dropout_rate)
        # else:
        #     self.cycle_encoder = LSTMEncoder(self.modality_dim[self.output_modality], opt.embd_size, embd_method=opt.embd_method)
        
        # # cycle decoder
        # if opt.output_modality == 'A':
        #     A_layers = list(map(lambda x: int(x), opt.A_layers.split(','))) + opt.embd_size
        #     self.cycle_decoder = FcEncoder(opt.embd_size, A_layers, dropout=opt.dropout_rate)
        # else:
        #     self.cycle_decoder = LSTMSeqDecoder(opt.embd_size, self.modality_dim[self.input_modality], 
        #                                 self.seq_len[self.input_modality])

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
            self.cur_iters = 0
        
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
        modality_name = {'A': 'acoustic', 'V': 'visual', 'L': 'lexical'}
        self.input_data = input[modality_name[self.input_modality]].float().to(self.device)
        self.output_data = input[modality_name[self.output_modality]].float().to(self.device)
        self.label = input['label'].to(self.device)
    
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.embd = self.netenc(self.input_data)
        self.recon = self.netdec(self.embd)
        self.logits, _ = self.netC(self.embd)
        self.pred = F.softmax(self.logits, dim=1)

    def backward(self):
        """Calculate the loss for back propagation"""
        # calc CE_loss 
        self.loss_CE = self.ce_weight * self.criterion_ce(self.logits, self.label)
        self.loss_mse = self.mse_weight * self.criterion_mse(self.recon, self.output_data)
        # self.loss_cycle = self.cycle_weight * self.criterion_mse(self.feat_fusion_miss.detach(), self.feat_cycle)
        loss = self.loss_CE + self.loss_mse # + self.loss_cycle
        self.loss_CE.backward()
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