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

'''
    Model for input [A Z Z] reconstruct -> [Z, V, L]
'''
class SimpleMissMapModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # parser.set_defaults(no_dropout=True)
        parser.add_argument('--mapping_layers', type=str, default='128,128,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--dropout_rate', type=float, default=0.2, help='rate of dropout')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in DNN')
        parser.add_argument('--miss_data_iters', type=int, default=4, help='# of iters train using missing data')
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        self.loss_names = ['MSE']#, 'map', 'MSE_A2L', 'MSE_L2A',]
        self.model_names = ['AE']
        # define networks
        mapping_layers = list(map(lambda x: int(x), opt.mapping_layers.split(',')))
        self.netAE = SimpleFcAE(128*3, mapping_layers, opt.dropout_rate, opt.bn)
        # define training settings 
        if self.isTrain:
            self.criterion_mse = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.cur_iters = 0
            self.miss_data_iters = opt.miss_data_iters
            self.miss2_rate = 0.5

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
        self.visual = input['visual'].float().cuda()
        self.lexical = input['lexical'].float().cuda()
        self.label = input['label'].cuda()
        if self.isTrain and self.cur_iters % (self.miss_data_iters+1) == 0:
            batch_size = self.acoustic.size(0)
            self.miss_index = torch.zeros([batch_size]).long().to(self.device)
            self.miss_index_matrix = torch.zeros([batch_size, 3]).to(self.device)
            self.miss_index = self.miss_index.random_(0, 3)
            self.miss_index = self.miss_index_matrix.scatter_(1, self.miss_index.unsqueeze(1), 1).long()
            self.miss_index[:int((1-self.miss2_rate) * batch_size)] = -1 * \
                            (self.miss_index[:int((1-self.miss2_rate) * batch_size)] - 1) # 前(1-self.miss2_rate)的数据取反, 

            self.acoustic_miss = self.acoustic * self.miss_index[:, 0].view([batch_size, 1]).float()
            self.lexical_miss = self.lexical * self.miss_index[:, 1].view([batch_size, 1]).float()
            self.visual_miss = self.visual * self.miss_index[:, 2].view([batch_size, 1]).float()
            # reverse data
            self.miss_index_reverse = -1 * (self.miss_index - 1)
            self.acoustic_reverse = self.acoustic * self.miss_index_reverse[:, 0].unsqueeze(-1).float()
            self.lexical_reverse = self.lexical * self.miss_index_reverse[:, 1].unsqueeze(-1).float()
            self.visual_reverse = self.visual * self.miss_index_reverse[:, 2].unsqueeze(-1).float()
            '''check'''
            '''
            for i, (a, l, v, ar, lr, vr, mi, mr, aa, ll, vv) in enumerate(zip(self.acoustic_miss, self.lexical_miss, self.visual_miss, \
                self.acoustic_reverse, self.lexical_reverse, self.visual_reverse, 
                self.miss_index, self.miss_index_reverse,
                self.acoustic, self.lexical, self.visual)):
            
                print('\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(i, 
                    torch.sum(aa).long().item(), torch.sum(ll).long().item(), torch.sum(vv).long().item(),
                    torch.sum(a).long().item(), torch.sum(l).long().item(), torch.sum(v).long().item(), 
                    torch.sum(ar).long().item(), torch.sum(lr).long().item(), torch.sum(vr).long().item(), 
                    mi.cpu().numpy(), mr.cpu().numpy()))
            '''
        else:
            self.acoustic_miss = self.acoustic_reverse = self.acoustic
            self.visual_miss = self.visual_reverse = self.visual
            self.lexical_miss = self.lexical_reverse = self.lexical
        
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.recon, self.latent = self.netAE(torch.cat([self.acoustic_miss, self.visual_miss, self.lexical_miss], dim=-1))
        self.T_embd = torch.cat([self.acoustic_reverse, self.visual_reverse, self.lexical_reverse], dim=-1)
        
    def backward(self):
        self.loss_MSE = self.criterion_mse(self.recon, self.T_embd)
        self.loss_MSE.backward()

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()   
        self.optimizer.zero_grad()  
        self.backward()
        self.optimizer.step()
        self.cur_iters += 1
