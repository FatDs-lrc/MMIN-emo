import torch
import os
import copy
import json
import torch.nn.functional as F
from .base_model import BaseModel
from .early_fusion_multi_model import EarlyFusionMultiModel
# from .networks.classifier import LSTMClassifier, FcClassifier, FcClassifier_nobn, EF_model_AL
from .networks.autoencoder import ResidualAE, ResidualUnetAE
from .networks.tools import init_net, MidLayerFeatureExtractor
from .utils.config import OptConfig

class ResidualAEModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # parser.set_defaults(no_dropout=True)
        parser.add_argument('--input_dim_a', type=int, default=128, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=128, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=128, help='visual input dim')
        parser.add_argument('--output_dim', type=int, default=4, help='output dim')
        parser.add_argument('--dropout_rate', type=float, default=0.2, help='rate of dropout')
        parser.add_argument('--AE_layers', type=str, default='128,32,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--n_blocks', type=int, default=3, help='number of AE blocks')
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--modality', type=str, help='which modality to use')
        parser.add_argument('--bn', action='store_true', help='whether add bn layer')
        parser.add_argument('--miss2_rate', type=float, default=0.2, help='# for miss_num=mix, probability of data which misses 2 modality')
        parser.add_argument('--teacher_path', type=str, default='None')
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
        self.teacher_path = os.path.join(opt.teacher_path, str(opt.cvNo))
        self.teacher_config_path = os.path.join(opt.teacher_path, 'train_opt.conf')
        self.teacher_config = self.load_from_opt_record(self.teacher_config_path)
        self.teacher_config.isTrain = False                             # teacher model should be in test mode
        self.teacher_config.gpu_ids = opt.gpu_ids                       # set gpu to the same
        self.teacher_model = EarlyFusionMultiModel(self.teacher_config)
        self.teacher_model.cuda()
        self.teacher_model.load_networks_cv(self.teacher_path)
        self.teacher_model.eval()
        self.loss_names = ['recon', 'CE']
        self.model_names = ['AE'] 
        fusion_layers = list(map(lambda x: int(x), opt.AE_layers.split(',')))
        input_dim = opt.input_dim_a + opt.input_dim_l + opt.input_dim_v
        self.netAE = ResidualUnetAE(fusion_layers, opt.n_blocks, input_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        self.netC = copy.deepcopy(self.teacher_model.netC)
        # self.set_requires_grad(self.netC, False)
        if self.isTrain:
            self.criterion_recon = torch.nn.MSELoss()
            self.criterion_CE = torch.nn.CrossEntropyLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.miss2_rate = opt.miss2_rate
            
        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
    
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        # for teacher, raw data no modality missing
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
            # default mix
            self.miss_index[:int((1-self.miss2_rate) * batch_size)] = -1 * \
                            (self.miss_index[:int((1-self.miss2_rate) * batch_size)] - 1) # 前(1-self.miss2_rate)的数据取反, 

            self.acoustic_miss = self.acoustic * self.miss_index[:, 0].unsqueeze(-1).float()
            self.lexical_miss = self.lexical * self.miss_index[:, 1].unsqueeze(-1).float()
            self.visual_miss = self.visual * self.miss_index[:, 2].unsqueeze(-1).float()
        else:
            self.miss_index = input['miss_index'].to(self.device)
            self.acoustic_miss = self.acoustic
            self.visual_miss = self.visual
            self.lexical_miss = self.lexical

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.feat_raw = torch.cat([self.acoustic_miss, self.lexical_miss, self.visual_miss], dim=-1)
        self.recon_feat, _ = self.netAE(self.feat_raw)        
        # print(self.mask.size())
        # if not self.isTrain:
        #     self.recon_A = self.recon_feat[:, :128]
        #     self.recon_L = self.recon_feat[:, 128:256]
        #     self.recon_V = self.recon_feat[:, 256:]
        #     self.mask = (-(self.miss_index.long() - 1)).float()
        #     self.recon_A = self.recon_A * self.mask[:, 0].unsqueeze(-1) + self.acoustic_miss
        #     self.recon_L = self.recon_L * self.mask[:, 1].unsqueeze(-1) + self.lexical_miss
        #     self.recon_V = self.recon_V * self.mask[:, 2].unsqueeze(-1) + self.visual_miss
        #     self.recon_feat = torch.cat([self.recon_A, self.recon_L, self.recon_V], dim=-1)
        #     print((self.recon_A==self.acoustic_miss).any())
        #     print((self.recon_L==self.lexical).any())
        #     print((self.recon_V==self.visual).any())
        self.logits, _ = self.netC(self.recon_feat)
        # self.logits, _ = self.netC(self.feat_raw)
        self.pred = F.softmax(self.logits, dim=-1)
   
    def backward(self):
        """Calculate the loss for back propagation"""
        # print(self.label)
        self.loss_recon = self.criterion_recon(self.recon_feat, self.feat_raw) * 0.3
        self.loss_CE = self.criterion_CE(self.logits, self.label) 
        loss = self.loss_recon + self.loss_CE
        loss.backward()

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()   
        # backward
        self.optimizer.zero_grad()  
        self.backward()            
        self.optimizer.step() 
    
    def load_from_opt_record(self, file_path):
        opt_content = json.load(open(file_path, 'r'))
        opt = OptConfig()
        opt.load(opt_content)
        return opt
