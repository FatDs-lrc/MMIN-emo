import torch
import os
import torch.nn.functional as F
from .base_model import BaseModel
from .networks.self_modules.fc_encoder import FcEncoder
from .networks.self_modules.lstm_encoder import LSTMEncoder
from .networks.self_modules.textcnn_encoder import TextCNN
from .networks.classifier import FcClassifier
from .networks.tools import init_net

class LateFusionMultiModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # parser.set_defaults(no_dropout=True)
        parser.add_argument('--input_dim_a', type=int, default=1582, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=384, help='lexical input dim')
        parser.add_argument('--mid_layers_a', type=str, default='256,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--hidden_size', default=128, type=int, help='rnn hidden layer')
        parser.add_argument('--embd_size_v', default=128, type=int, help='embedding size for v')
        parser.add_argument('--embd_size_l', default=128, type=int, help='embedding size for l')
        parser.add_argument('--embd_method', default='last', type=str, help='LSTM encoder embd function')
        parser.add_argument('--fusion_size', type=int, default=384, help='fusion model fusion size')
        parser.add_argument('--mid_layers_fusion', type=str, default='128,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
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
        # self.teacher_path = os.path.join(opt.teacher_path, f'cvNo{opt.cvNo}')
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        self.loss_names = ['CE']
        self.modality = opt.modality

        # acoustic model
        if 'A' in self.modality:
            self.model_names += ['A', 'A_C']
            A_layers = list(map(lambda x: int(x), opt.mid_layers_a.split(',')))
            self.netA = FcEncoder(opt.input_dim_a, A_layers[:-1], opt.dropout_rate, opt.bn)
            self.netA_C = FcClassifier(A_layers[-2], [A_layers[-1]], opt.output_dim, dropout=opt.dropout_rate)

        # lexical model
        if 'L' in self.modality:
            self.model_names += ['L', 'L_C']
            self.netL = TextCNN(opt.input_dim_l, opt.embd_size_l)
            self.netL_C = FcClassifier(opt.embd_size_l, [opt.fusion_size], opt.output_dim, dropout=opt.dropout_rate)
            
        # visual model
        if 'V' in self.modality:
            self.model_names += ['V', 'V_C']
            self.netV = LSTMEncoder(opt.input_dim_v, opt.hidden_size, opt.embd_method)
            self.netV_C = FcClassifier(opt.embd_size_v, [opt.fusion_size], opt.output_dim, dropout=opt.dropout_rate)
            # self.netV = TextCNN(opt.input_dim_v, opt.embd_size_v)
            
        if self.isTrain:
            self.criterion_ce = torch.nn.NLLLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
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
        self.visual = input['visual'].float().cuda()
        self.length_l = input['l_length'].long().cuda()
        self.length_v = input['v_length'].long().cuda()
        self.label = input['label'].cuda()
        # self.label = self.label.argmax(dim=1)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        final_pred = []
        if 'A' in self.modality:
            self.embd_A = self.netA(self.acoustic)
            self.logits_A, _ = self.netA_C(self.embd_A)
            self.pred_A = F.softmax(self.logits_A, dim=-1)
            final_pred.append(self.pred_A)

        if 'L' in self.modality:
            self.embd_L = self.netL(self.lexical)
            self.logits_L, _ = self.netL_C(self.embd_L)
            self.pred_L = F.softmax(self.logits_L, dim=-1)
            final_pred.append(self.pred_L)
        
        if 'V' in self.modality:
            self.embd_V = self.netV(self.visual)
            self.logits_V, _ = self.netV_C(self.embd_V)
            self.pred_V = F.softmax(self.logits_V, dim=-1)
            final_pred.append(self.pred_V)
        
        # late fusion
        self.pred = torch.stack(final_pred)
        self.pred = torch.mean(self.pred, dim=0)
   
    def backward(self):
        """Calculate the loss for back propagation"""
        self.loss_CE = self.criterion_ce(torch.log(self.pred), self.label)
        self.loss_CE.backward()
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
