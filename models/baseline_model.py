import torch
import os
import torch.nn.functional as F
from .base_model import BaseModel
from .teacher_networks import TeacherModel
from .networks.classifier import LSTMClassifier, FcClassifier, FcClassifier_nobn, EF_model_AL
from .networks.tools import init_net, MidLayerFeatureExtractor
from .networks.mmd_loss import MMD_loss

class BaselineModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # parser.set_defaults(no_dropout=True)
        parser.add_argument('--mid_layers', type=str, default='256,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--input_dim_a', type=int, default=1582, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--output_dim', type=int, default=4, help='output dim')
        parser.add_argument('--dropout_rate', type=float, default=0.2, help='rate of dropout')
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--modality', type=str, help='which modality to use')
        parser.add_argument('--bn', type=bool, default=True, help='whether add bn layer')
        parser.add_argument('--hidden_size', default=256, type=int, help='lstm hidden layer')
        parser.add_argument('--fc1_size', default=128, type=int, help='lstm embedding size')
        parser.add_argument('--fusion_size', type=int, default=128, help='fusion model fusion size')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        self.loss_names = ['CE']
        self.modality = opt.modality
        # acoustic modal
        if self.modality == 'acoustic':
            self.model_names = ['A'] 
            layers = list(map(lambda x: int(x), opt.mid_layers.split(',')))
            if opt.bn:
                self.netA = init_net(FcClassifier(opt.input_dim_a, layers, opt.output_dim, dropout=opt.dropout_rate), 
                                opt.init_type, opt.init_gain, gpu_ids=opt.gpu_ids)
            else:
                self.netA = init_net(FcClassifier_nobn(opt.input_dim_a, layers, opt.output_dim, dropout=opt.dropout_rate), 
                                opt.init_type, opt.init_gain, gpu_ids=opt.gpu_ids)

        # lexical modal
        elif self.modality == 'lexical':
            self.model_names = ['L'] 
            self.netL = init_net(LSTMClassifier(1024, opt.hidden_size, opt.fc1_size, opt.output_size, opt.dropout_rate),
                                opt.init_type, opt.init_gain, gpu_ids=opt.gpu_ids)

        elif self.modality == 'A+L':
            self.model_names = ['AL'] 
            layers = list(map(lambda x: int(x), opt.mid_layers.split(',')))
            out_dim_a = layers[-1]
            out_dim_l = opt.fc1_size
            self.netA = FcClassifier(opt.input_dim_a, layers, opt.output_dim, dropout=opt.dropout_rate)
            self.netL = LSTMClassifier(opt.input_dim_l, opt.hidden_size, opt.fc1_size, opt.output_dim, opt.dropout_rate)
            self.netAL = init_net(EF_model_AL(self.netA, self.netL, out_dim_a, out_dim_l, 
                                opt.fusion_size, opt.output_dim, opt.dropout_rate),
                                opt.init_type, opt.init_gain, gpu_ids=opt.gpu_ids)

        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.Adam(self.netA.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
           
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
        self.l_mask = input['l_mask'].float().cuda()
        # self.lexical = input['lexical'].float().to(self.device)
        self.label = input['label'].to(self.device)
        # self.label = self.label.argmax(dim=1)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.modality == 'acoustic':
            self.logits, self.feat = self.netA(self.acoustic)
        elif self.modality == 'lexical':
            self.logits, self.feat = self.netL(self.lexical, self.l_mask)
        elif self.modality == 'A+L':
            self.logits, self.feat = self.netAL(self.acoustic, self.lexical, self.l_mask)
        
        self.pred = F.softmax(self.logits, dim=-1)
   
    def backward(self):
        """Calculate the loss for back propagation"""
        # print(self.label)
        self.loss_CE = self.criterion_ce(self.pred, self.label)
        self.loss_CE.backward()

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()   
        # backward
        self.optimizer.zero_grad()  
        self.backward()            
        self.optimizer.step() 
