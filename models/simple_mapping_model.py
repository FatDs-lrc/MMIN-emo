import torch
import os
import torch.nn.functional as F
from .base_model import BaseModel
from .networks.self_modules.fc_encoder import FcEncoder
from .networks.autoencoder import SimpleFcAE, BaseAutoencoder
from .networks.self_modules.lstm_encoder import LSTMEncoder
from .networks.self_modules.textcnn_encoder import TextCNN
from .networks.classifier import FcClassifier
from .networks.tools import init_net

'''
Model for L->A autoencoder using deep utt level feature.
'''
class SimpleMappingModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # parser.set_defaults(no_dropout=True)
        parser.add_argument('--miss_data_iters', type=int, default=4, help='# of iters train using missing data')
        parser.add_argument('--mapping_layers', type=str, default='128,128,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--dropout_rate', type=float, default=0.2, help='rate of dropout')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in DNN')
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--model_type', type=str, help='input modality 2 output modality')
        parser.add_argument('--mse_weight', type=float, default=0.2, help='mse weight')
        parser.add_argument('--cycle_weight', type=float, default=0.2, help='cycle_loss weight')
        
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        self.loss_names = ['CE', 'MSE', 'cycle']#, 'map', 'MSE_A2L', 'MSE_L2A',]
        self.model_names = ['AE', 'C', 'C_fusion', 'AE_cycle']
        # define networks
        mapping_layers = list(map(lambda x: int(x), opt.mapping_layers.split(',')))
        # self.netAE = SimpleFcAE(128, mapping_layers, 0, opt.bn)
        # self.netAE_cycle = SimpleFcAE(128, mapping_layers, 0, opt.bn)
        self.netAE = BaseAutoencoder()
        self.netAE_cycle = BaseAutoencoder()
        self.netC = FcClassifier(32 + 128, [128], 4, dropout=0.5, use_bn=False)
        self.netC_fusion = FcClassifier(200, [128], 4, dropout=0.5, use_bn=False)
        self.input_m = opt.model_type[0]
        self.output_m = opt.model_type[-1]
        # define training settings 
        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            self.criterion_mse = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.mse_weight = opt.mse_weight
            self.cycle_weight = opt.cycle_weight

        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.acoustic = input['acoustic'].float().to(self.device)
        self.visual = input['visual'].float().to(self.device)
        self.lexical = input['lexical'].float().to(self.device)
        self.label = input['label'].cuda()
        modality_data_lookup = {
            'A': self.acoustic,
            'V': self.visual,
            'L': self.lexical
        }
        
        self.input_modality_data = modality_data_lookup[self.input_m]
        self.output_modality_data = modality_data_lookup[self.output_m]
        
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.recon, self.latent = self.netAE(self.input_modality_data)
        self.recon_cycle, self.latent_cycle = self.netAE_cycle(self.recon)
        self.logits, self.feat = self.netC(torch.cat([self.input_modality_data, self.latent], dim=-1))
        # self.logits_fusion, self.feat_fusion = self.netC_fusion(torch.cat([self.input_modality_data, self.latent], dim=-1))
        self.pred = F.softmax(self.logits)

    def backward(self):
        self.loss_CE = self.criterion_ce(self.logits, self.label)
        # self.loss_CE_fusion = self.criterion_ce(self.logits_fusion, self.label)
        self.loss_MSE = self.mse_weight * self.criterion_mse(self.recon, self.output_modality_data)
        self.loss_cycle = self.cycle_weight * self.criterion_mse(self.input_modality_data, self.recon_cycle)
        loss = self.loss_CE + self.loss_MSE + self.loss_cycle
        loss.backward()

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()   
        self.optimizer.zero_grad()  
        self.backward()
        self.optimizer.step()
