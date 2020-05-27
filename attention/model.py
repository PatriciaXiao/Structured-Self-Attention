import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
 
class StructuredSelfAttention(torch.nn.Module):
    """
    The class is an implementation of the paper A Structured Self-Attentive Sentence Embedding including regularization
    and without pruning. Slight modifications have been done for speedup
    """
   
    def __init__(self,batch_size,lstm_hid_dim,d_a,r,max_len, num_layers=1, emb_dim=100,vocab_size=None,use_pretrained_embeddings = False,embeddings=None,_type=0,n_classes = 1):
        """
        Initializes parameters suggested in paper
 
        Args:
            batch_size  : {int} batch_size used for training
            lstm_hid_dim: {int} hidden dimension for lstm
            d_a         : {int} hidden dimension for the dense layer
            r           : {int} attention-hops or attention heads
            max_len     : {int} number of lstm timesteps
            emb_dim     : {int} embeddings dimension
            vocab_size  : {int} size of the vocabulary
            use_pretrained_embeddings: {bool} use or train your own embeddings
            embeddings  : {torch.FloatTensor} loaded pretrained embeddings
            type        : [0,1] 0-->binary_classification 1-->multiclass classification
            n_classes   : {int} number of classes
 
        Returns:
            self
 
        Raises:
            Exception
        """
        super(StructuredSelfAttention,self).__init__()
       
        self.embeddings = torch.nn.Embedding(vocab_size,emb_dim,padding_idx=0)
        self.lstm = torch.nn.LSTM(emb_dim,lstm_hid_dim,num_layers,batch_first=True,bidirectional=True)
        self.linear_first = torch.nn.Linear(lstm_hid_dim*2,d_a) # W_{s_1}
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a,r) # W_{s_2}
        self.linear_second.bias.data.fill_(0)
        self.n_classes = n_classes
        self.linear_final = torch.nn.Linear(2*lstm_hid_dim,self.n_classes) 
        self.batch_size = batch_size       
        self.max_len = max_len
        self.lstm_hid_dim = lstm_hid_dim
        self.hidden_state = self.init_hidden()
        self.regularization = r
        self.type = _type
       
        
    def softmax(self,input, axis=1):
        """
        Softmax applied to axis=n
 
        Args:
           input: Tensor, input on which softmax is to be applied
           axis : int axis on which softmax is to be applied
 
        Returns:
            softmaxed tensors

        """
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])

        soft_max_2d = F.softmax(input_2d, dim=1) # dim=?: originally unspecified
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)
       
        
    def init_hidden(self):
        return  torch.zeros(2,self.batch_size,self.lstm_hid_dim), \
                torch.zeros(2,self.batch_size,self.lstm_hid_dim)
       
        
    def forward(self,x):
        embeddings = self.embeddings(x)       
        outputs, self.hidden_state = self.lstm(embeddings.view(self.batch_size,self.max_len,-1),self.hidden_state)       
        x = torch.tanh(self.linear_first(outputs))       
        x = self.linear_second(x)       
        x = self.softmax(x,1)       
        attention = x.transpose(1,2)  # A     
        sentence_embeddings = attention@outputs   # attended outout and the original output    
        avg_sentence_embeddings = torch.sum(sentence_embeddings,axis=1)/self.regularization
       
        if not bool(self.type):
            output = torch.sigmoid(self.linear_final(avg_sentence_embeddings))
           
            return output,attention
        else:
            return F.log_softmax(self.linear_final(avg_sentence_embeddings)),attention
       
	   
	#Regularization
    def l2_matrix_norm(self,m):
        """
        Frobenius norm calculation
 
        Args:
           m: {Variable} ||AAT - I||
 
        Returns:
            regularized value

        """
        return torch.sum(torch.sum(torch.sum(m**2,1),1)**0.5).type(torch.DoubleTensor)

