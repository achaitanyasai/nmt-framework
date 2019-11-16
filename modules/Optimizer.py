import torch.optim as optim
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from structs import *

'''
Same as OpenNMT Optim.py
https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Optim.py
'''

class Optimizer(object):
    def __init__(self, optimizer_type, lrate, max_grad_norm,
                 lrate_decay=1, start_decay_at=None,
                 decay_lrate_steps=50, beta1=0.9, beta2=0.999):
        self.optimizer_type = optimizer_type
        self.lrate = lrate
        self.original_lrate = lrate
        self.max_grad_norm = max_grad_norm
        self.lrate_decay = lrate_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self.decay_lrate_steps = decay_lrate_steps
        self._step = 0
        self.betas = (beta1, beta2)
        self.last_decayed = 0
        self.last_loss = 1000000000000000.0

    def set_parameters(self, params):
        self.params = [p for p in params if p.requires_grad]
        if self.optimizer_type == 'sgd':
            logger.info('SGD, lr: %.6f' % self.lrate)
            self.optimizer = optim.SGD(self.params, lr=self.lrate)
        elif self.optimizer_type == 'adam':
            logger.info('Adam, lr: %.6f' % self.lrate)
            self.optimizer = optim.Adam(self.params, lr=self.lrate, betas=self.betas, eps=1e-9)
        else:
            raise RuntimeError("Invalid optim method: " + self.optimizer_type)

    def _set_rate(self, lrate):
        self.lrate = lrate
        self.optimizer.param_groups[0]['lr'] = self.lrate

    def step(self):
        """Update the model parameters based on current gradients.
        Optionally, will employ gradient modification or update learning
        rate.
        """
        self._step += 1

        # Decay method used in tensor2tensor.
        # if self.decay_method == "noam":
        #     self._set_rate(
        #         self.original_lrate *
        #         (self.model_size ** (-0.5) *
        #          min(self._step ** (-0.5),
        #              self._step * self.warmup_steps**(-1.5))))

        if self.max_grad_norm:
            clip_grad_norm(self.params, self.max_grad_norm)
        self.optimizer.step()

    def update_learning_rate(self, cur_loss, step):
        """
        Decay learning rate if val perf does not improve
        or we hit the start_decay_at limit.
        """
        if step >= self.start_decay_at:
            if step - self.last_decayed >= self.decay_lrate_steps:
                self.last_decayed = step
                self.lrate = self.lrate * self.lrate_decay
                self.optimizer.param_groups[0]['lr'] = self.lrate

        # if cur_loss > self.last_loss and step >= self.start_decay_at:
        #     self.start_decay = True
        #     self.last_decayed = step
        # else:
        #     if self.start_decay_at is not None and step >= self.start_decay_at:
        #         if step - self.last_decayed >= self.decay_lrate_steps:
        #             self.start_decay = True
        #             self.last_decayed = step
        #         else:
        #             self.start_decay = False
        #     else:
        #         self.start_decay = False
        # if self.start_decay:
        #     if step - self.last_decayed >= self.decay_lrate_steps:
        #         self.lrate = self.lrate * self.lrate_decay
        #         logger.info('Decaying learning rate to %.6f' % self.lrate)
        #
        # self.last_loss = cur_loss

