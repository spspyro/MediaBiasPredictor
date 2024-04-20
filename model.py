import torch
import torch.nn as nn

class MediaBiasNN(nn.Module):
    def __init__(self,
                 hidden_dim: list,
                 model_type='baseline',
                 lstm_layers=1,
                 lstm_hidden_dim=1024,
                 upgrade=False,
                 freeze_bert=True):
        '''
        Create Media Bias Nerual Network.
        There are 4 types of model that can be generated from this class.
        Baseline, Baseline+, RNN, RNN+.
        Args:
            hidden_dim:
                list of hidden dimension for the last linear head.
            model_type:
                "baseline" or "rnn"
                BASELINE uses average sum to generate article embed
                RNN uses LSTM to generate article embed
            lstm_layers:
                number of stack layer in lstm cell
            lstm_hidden_dim: 
                hidden dim for lstm cell
            upgrade: 
                upgrade the baseline to plus model by adding features
            freeze_bert: 
                freeze bert or not
        '''
        super().__init__()
        # some variable usefule later
        self.num_class = 3
        self.plus = upgrade
        if model_type=="baseline":
            self.out_dim = 768
        else:
            self.out_dim = lstm_hidden_dim
        # create loss fucntion
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        # create and init classification head
        self.cls_head = self._create_head(hidden_dim)
        self._init_head()
        # create backbone
        #self.bert = BertModel.from_pretrained("bert-base-uncased")
        #if freeze_bert:
        #    self._freeze_backbone()
        #else:
        #    self._unfreeze_backbone()
        # create lstm if neccessary
        self.model_type = model_type
        if model_type=="rnn":
            self.rnn = self._create_rnn(lstm_hidden_dim, lstm_layers)
        

    def _create_head(self, hidden_dim):
        '''
        Create classification head based on hidden_dim.
        Args:
            hidden_dim:
                List of int indicating hidden dim for each of the linear layer
                in the classifiation head.
        Return:
            nn.Module: classifier head.
        '''
        head = []
        hidden_dim = [self.out_dim] + hidden_dim
        for h in range(1, len(hidden_dim)):
            head.append(nn.Linear(hidden_dim[h-1], hidden_dim[h]))
            head.append(nn.ReLU())
        head.append(nn.Linear(hidden_dim[-1], self.num_class))
        head = nn.Sequential(*head)
        return head

    def _init_head(self):
        '''
        Initialize the prediction head.
        '''
        for layer in self.cls_head.modules():
            #print(layer)
            if isinstance(layer, nn.Linear):
                torch.nn.init.normal_(
                    layer.weight, mean=0, std=1)
                torch.nn.init.constant_(
                    layer.bias, 0)

    def _create_rnn(self, hidden_size, num_layers):
        '''
        Create LSTM for RNN type model.
        Args:
            hidden_size:
                Hidden size of LSTM cell
            num_layers:
                Number of stacked LSTM cell
        Return:
            nn.Module: lstm layer
        '''
        lstm = nn.LSTM(768, hidden_size, num_layers)
        return lstm

    """
    def _freeze_backbone(self):
        '''
        Freeze bert.
        '''
        for params in self.bert.parameters():
            params.requires_grad = False
    
    def _unfreeze_backbone(self):
        '''
        Unfreeze bert.
        '''
        for params in self.bert.parameters():
            params.requires_grad = True
    """

    def _add_feature(self, content_embedding, article):
        '''
        Args:
            content_embedding:
                content embedding
            article:
                Article ditionary that contain other meta data
                such as title, publisher, etc.
        Return:
            tensor: modified content embedding
        '''
        # TODO Extract additional feature from article
        # and concat it to embedding. OPTIONAL.
        return content_embedding

    def _remove_padding(self, token_embed, attn_mask):
        '''
        Remove padding embedding from token embedding.
        Args:
            token_embed:
                shape Num_sent x Max_length x Token_embed
            attn_mask:
                shape Num_sent x Max_length
        Return:
            tensor: token embedding without padding embedding
        '''
        return token_embed[attn_mask == 1]
    
    def _get_content_embedding(self, token_embedding):
        '''
        Take token embedding from content and return content embedding.
        Args:
            token_embedding:
                Tensor of shape N x embed_dim, where N is the number of none-pad
                token and embed_dim is the embedding dimension outputed by bert.
        Return:
            tensor: content representation, differ based on model type.
        '''
        if self.model_type=="baseline":
            content_embedding = token_embedding.mean(0)
        else:
            output, (h_n, c_n) = self.rnn(token_embedding[None,:,:])
            #print(output.shape, h_n.shape, c_n.shape)
            content_embedding = h_n[-1].mean(dim=0)
            #print(content_embedding.shape)
        return content_embedding
    
    def forward(self, x, y = None):
        '''
        Args:
            x: List of dictionary that contain from content.
                input_ids: 
                    token ids from content sentence
                    shape: number_sent x max_length_sent
                token_type_ids: 
                    *ignore for now.
                    shape: number_sent x max_length_sent
                attention_mask:
                    inidicate which part of the input ids are padding.
                    could be useful later.
                    shape: number_sent x max_length_sent
            y: target
                optional, depends on the model mode.
        Return:
            tensor: loss if model.mode == training
                    prediciton if model.mode == eval
        '''
        pred = []
        for token_embedding in x:
            content_embedding = self._get_content_embedding(token_embedding)
            #if self.plus:
            #    content_embedding = self._add_feature(content_embedding, article)
            pred.append(self.cls_head(content_embedding))
        if not self.training:
            return torch.stack(pred)
        else:
            pred = torch.stack(pred, dim=0)
            loss = self.loss(pred, y)
            return loss














