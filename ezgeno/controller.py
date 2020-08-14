import torch
import torch.nn.functional as F
import utils

class Controller(torch.nn.Module):
    def __init__(self, args,deepbind_num_of_conv_choices,deepbind_layers,score_num_of_conv_choices=0,score_layers=0):
        torch.nn.Module.__init__(self)
        self.args = args
        self.controller_hid = 100

        self.deepbind_num_of_conv_choices=deepbind_num_of_conv_choices
        self.deepbind_layers=deepbind_layers
        self.score_num_of_conv_choices=score_num_of_conv_choices
        self.score_layers=score_layers

        self.encoder = torch.nn.Embedding(deepbind_num_of_conv_choices, self.controller_hid)
        self.lstm = torch.nn.LSTMCell(self.controller_hid, self.controller_hid)

        self.decoders = []

        for i in range(self.deepbind_layers):
            decoder = torch.nn.Linear(self.controller_hid, self.deepbind_num_of_conv_choices)
            self.decoders.append(decoder)
            decoder = torch.nn.Linear(self.controller_hid, i+1)
            self.decoders.append(decoder)
            
        for i in range(self.score_layers):
            decoder = torch.nn.Linear(self.controller_hid, self.score_num_of_conv_choices)
            self.decoders.append(decoder)
            decoder = torch.nn.Linear(self.controller_hid, i+1)
            self.decoders.append(decoder)    

        self._decoders = torch.nn.ModuleList(self.decoders)

        self.reset_parameters()
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)

        def _get_default_hidden(key):
            return utils.get_variable(
                torch.zeros(key, self.controller_hid),
                self.args.cuda,
                requires_grad=False)

        self.static_inputs = utils.keydefaultdict(_get_default_hidden)

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.decoders:
            decoder.bias.data.fill_(0)

    def forward(self,
                inputs,
                hidden,
                block_idx,
                is_embed,
                is_train=True):
        if not is_embed:
            embed = self.encoder(inputs)
        else:
            embed = inputs

        hx, cx = self.lstm(embed, hidden)
        logits = self.decoders[block_idx](hx)

        logits /= 5

        if is_train:
            logits = (2.5*torch.tanh(logits))

        return logits, (hx, cx)

    def sample(self, batch_size=1, with_details=False, save_dir=None, is_train=True):

        inputs = self.static_inputs[batch_size]
        hidden = self.static_init_hidden[batch_size]

        entropies = []
        log_probs = []
        policy = []
        
        total_layers=self.deepbind_layers+self.score_layers
        for block_idx in range(total_layers*2):###
            logits, hidden = self.forward(inputs,
                                          hidden,
                                          block_idx,
                                          is_embed=(block_idx == 0),
                                        is_train=(True and is_train))

            probs = F.softmax(logits, dim=-1)

            log_prob = F.log_softmax(logits, dim=-1)

            entropy = -(log_prob * probs).sum(1, keepdim=False)
            
            if is_train:
                action = probs.multinomial(num_samples=1).data
            else:
                action = probs.argmax(dim=1, keepdim=True)
            policy.append(action.item())
            selected_log_prob = log_prob.gather(
                1, utils.get_variable(action, self.args.cuda, requires_grad=False))

            entropies.append(entropy)
            log_probs.append(selected_log_prob[:, 0])

            inputs = utils.get_variable(action[:, 0], self.args.cuda, requires_grad=False)
        

        if with_details:
            return policy, torch.cat(log_probs), torch.cat(entropies)

        return policy

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.controller_hid)
        return (utils.get_variable(zeros, self.args.cuda, requires_grad=False),
                utils.get_variable(zeros.clone(), self.args.cuda, requires_grad=False))
