
import torch
import os
import json
import torch.nn.functional as F
from .base_model import BaseModel
from .networks.self_modules.fc_encoder import FcEncoder
from .networks.self_modules.lstm_encoder import LSTMAutoencoder, LSTMEncoder
from .networks.self_modules.textcnn_encoder import TextCNN
from .networks.classifier import FcClassifier
from .networks.tools import init_net, MultiLayerFeatureExtractor
from .networks.self_modules.loss import MMD_loss, SoftCenterLoss, KLDivLoss_OnFeat
from .utils.config import OptConfig


class TranslationModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--input_modality', default='L', type=str, help='input existing modality')
        parser.add_argument('--aug_modality1', default='A', type=str, help='The first assist modality')
        parser.add_argument('--aug_modality2', default='None', type=str, help='The second assist modality')
        parser.add_argument('--hidden_size', default=128, type=int, help='rnn hidden layer')
        parser.add_argument('--embd_method', default='last', type=str, help='visual embedding method,last,mean or atten')
        parser.add_argument('--fusion_size', type=int, default=384, help='fusion model fusion size')
        parser.add_argument('--mid_layers_fusion', type=str, default='128,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--output_dim', type=int, default=4, help='output dim')
        parser.add_argument('--dropout_rate', type=float, default=0.2, help='rate of dropout')
        parser.add_argument('--lambda_ce', type=float, default=1.0, help='weight of ce loss')
        parser.add_argument('--lambda_cycle', type=float, default=0.1, help='weight of mse loss')
        parser.add_argument('--lambda_mse1', type=float, default=0.1, help='weight of mse loss')
        parser.add_argument('--lambda_mse2', type=float, default=0.1, help='weight of mse loss')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        self.loss_names = ['CE', 'MSE1', 'cycle']
        self.model_names = ['C', 'Emo', 'T1', '_cycle']
        # define network
        self.netEmo = LSTMEncoder(opt.hidden_size, opt.hidden_size, embd_method='maxpool', pool_len=22)
        fusion_layers = list(map(lambda x: int(x), opt.mid_layers_fusion.split(',')))
        self.netC = FcClassifier(opt.fusion_size, fusion_layers, opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        self.dims_lookup = {
            'L': 300,
            'V': 342,
            'A': 130
        }
        self.hidden_size = opt.hidden_size
        self.input_modality = opt.input_modality
        self.aug_modality1 = opt.aug_modality1
        self.aug_modality2 = opt.aug_modality2
        self.input_dim = self.dims_lookup[self.input_modality]
        self.target1_dim = self.dims_lookup[self.aug_modality1]
        self.netT1 = LSTMAutoencoder(self.input_dim, self.hidden_size, self.target1_dim)
        self.net_cycle = LSTMAutoencoder(self.target1_dim, self.hidden_size, self.input_dim)
        if self.aug_modality2 != 'None':
            assert self.aug_modality1 != self.aug_modality2
            self.target2_dim =  self.dims_lookup[self.aug_modality2]
            self.model_names.append('T2')
            self.loss_names.append('MSE2')
            self.netT2 = LSTMAutoencoder(self.hidden_size, self.hidden_size, self.target2_dim)
        
        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            self.criterion_mse = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.output_dim = opt.output_dim
            self.lambda_ce = opt.lambda_ce
            self.lambda_MSE1 = opt.lambda_mse1
            self.lambda_cycle = opt.lambda_cycle
            self.lambda_MSE2 = opt.lambda_mse2

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

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        input_lookup = {
            'A': self.acoustic,
            'V': self.visual,
            'L': self.lexical
        }
        self.input_feat = input_lookup[self.input_modality]
        self.aug1_target = input_lookup[self.aug_modality1]
        self.recon1, self.latent = self.netT1(self.input_feat)
        self.recon_cycle, _ = self.net_cycle(self.recon1)
        if self.aug_modality2 != "None":
            self.aug2_target = input_lookup[self.aug_modality2]
            self.recon2, self.latent = self.netT2(self.latent)
        
        # get model outputs
        self.feat = self.netEmo(self.latent)
        self.logits, _ = self.netC(self.feat)
        self.pred = F.softmax(self.logits, dim=-1)
        
    def backward(self):
        """Calculate the loss for back propagation"""
        self.loss_CE = self.criterion_ce(self.logits, self.label) * self.lambda_ce
        self.loss_MSE1 = self.criterion_mse(self.recon1, self.aug1_target) * self.lambda_MSE1
        self.loss_cycle = self.criterion_mse(self.recon_cycle, self.input_feat) * self.lambda_cycle
        loss = self.loss_CE + self.loss_MSE1 + self.loss_cycle

        if self.aug_modality2 != "None":
            self.loss_MSE2 = self.criterion_mse(self.recon2, self.aug2_target) * self.lambda_MSE2
            loss += self.loss_MSE2
        
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
