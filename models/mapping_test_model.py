import torch
import os
import torch.nn.functional as F
from .base_model import BaseModel
from .networks.self_modules.fc_encoder import FcEncoder
from .networks.autoencoder import SimpleFcAE
from .networks.self_modules.lstm_encoder import LSTMEncoder
from .networks.self_modules.textcnn_encoder import TextCNN
from .networks.classifier import FcClassifier
from .networks.tools import init_net

class MappingTestModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # parser.set_defaults(no_dropout=True)
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
        self.loss_names = ['CE', 'MSE']#, 'map', 'MSE_A2L', 'MSE_L2A',]
        self.model_names = ['A_C', 'L_C', 'C', 'A2L', 'L2A']
        # fusion classifier
        fusion_layers = list(map(lambda x: int(x), opt.mid_layers_fusion.split(',')))
        self.netC = FcClassifier(opt.fusion_size, fusion_layers, opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        # acoustic model
        self.netA_C = FcClassifier(128, fusion_layers, opt.output_dim, dropout=opt.dropout_rate)
        # lexical model
        self.netL_C = FcClassifier(128, fusion_layers, opt.output_dim, dropout=opt.dropout_rate)
        # net A2L
        mapping_layers = list(map(lambda x: int(x), opt.mapping_layers.split(',')))
        self.netA2L = SimpleFcAE(128, mapping_layers, opt.dropout_rate, opt.bn)
        # net L2A
        self.netL2A = SimpleFcAE(128, mapping_layers, opt.dropout_rate, opt.bn)
        # define training settings 
        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            self.criterion_mse = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters_cls = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters_cls, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
    
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
        # if not self.isTrain:
        #     self.miss_index = input['miss_index']
        
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        
        
        # self.LfromA, self.LfromA_latent = self.netA2L(self.acoustic.detach())
        # self.logits_AfromL, self.feat_AfromL = self.netA_C(self.AfromL)
        # self.logits_AfromL_L, self.feat_AfromL_L = self.netC(torch.cat([self.AfromL, self.lexical], dim=-1))
        # self.logits_A_LfromA, self.feat_A_LfromA = self.netC(torch.cat([self.acoustic, self.LfromA_latent], dim=-1))
        self.AfromL, self.AfromL_latent = self.netL2A(self.lexical)
        self.logits, self.feat = self.netC(self.AfromL_latent)
        self.pred = F.softmax(self.logits, dim=-1)
        
    def backward(self):
        # self.loss_CE_F = self.criterion_ce(self.logits_ef, self.label) * 0.5
        # self.loss_CE_F.backward(retain_graph=True)
        # self.loss_CE_fake = self.criterion_ce(self.logits_A_LfromA, self.label)
        # self.loss_CE_fake.backward(retain_graph=True)
        self.loss_CE = self.criterion_ce(self.logits, self.label)
        self.loss_MSE = self.criterion_mse(self.acoustic, self.AfromL)
        loss = self.loss_CE + self.loss_MSE
        loss.backward()
        # self.loss_CE_A = self.criterion_ce(self.logits_A, self.label)
        # self.loss_CE_A.backward()

        # self.loss_CE_L = self.criterion_ce(self.logits_L, self.label)
        # self.loss_CE_L.backward()

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()   
        self.optimizer.zero_grad()  
        self.backward()
        self.optimizer.step()
