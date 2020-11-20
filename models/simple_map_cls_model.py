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
from .utils.load_pretrained import load_pretrained_model
from .simple_mapping_model import SimpleMappingModel

class SimpleMapClsModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # parser.set_defaults(no_dropout=True)
        parser.add_argument('--model_type', type=str, help='input and output modality')
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        self.loss_names = ['CE']#, 'map', 'MSE_A2L', 'MSE_L2A',]
        self.model_names = ['C']
        # define networks
        self.netC = FcClassifier(128, [128], 4, dropout=0.5, use_bn=False)
        # self.AE_model = load_pretrained_model(SimpleMappingModel, 'checkpoints/simple_map_{}_run1'.format(opt.model_type), opt.cvNo, opt.gpu_ids)
        # define training settings 
        if self.isTrain:
            self.criterion = torch.nn.CrossEntropyLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.model_type = opt.model_type
            self.input_modality = self.model_type[0]

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
        self.visual = input['visual'].float().cuda()
        self.label = input['label'].cuda()
        modality_data_lookup = {
            'A': self.acoustic,
            'V': self.visual,
            'L': self.lexical
        }
        self.input_data = modality_data_lookup[self.input_modality]
        
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # with torch.no_grad():
        #     self.recon, self.latent = self.AE_model.netAE(self.input_data)
        # self.reps = torch.cat([self.recon, self.input_data], dim=-1)
        # self.logits, self.feat = self.netC(self.reps)
        self.logits, self.feat = self.netC(self.input_data)
        self.pred = F.softmax(self.logits, dim=-1)
        
    def backward(self):
        self.loss_CE = self.criterion(self.logits, self.label)
        self.loss_CE.backward()

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()   
        self.optimizer.zero_grad()  
        self.backward()
        self.optimizer.step()
