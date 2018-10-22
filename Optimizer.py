#-----------------------------------------------------------------------------
# function to get optimizer
# scheduler class for optimizer parameter adjustment
#-----------------------------------------------------------------------------

import torch

def get_optimizer(Ps, encoder, decoder):
    #op=torch.optim.SGD([{'params': filter(lambda p:p.requires_grad, encoder.parameters()), 'lr':Ps["encoder_lr"], 'weight_decay':Ps["encoder_wd"]},
    #                    {'params': filter(lambda p:p.requires_grad, decoder.parameters()), 'lr':Ps["decoder_lr"], 'weight_decay':Ps["decoder_wd"]}],momentum=0.9)
    op=torch.optim.Adam([{'params': filter(lambda p:p.requires_grad, encoder.parameters()), 'lr':Ps["encoder_lr"], 'weight_decay':Ps["encoder_wd"]},
                         {'params': filter(lambda p:p.requires_grad, decoder.parameters()), 'lr':Ps["decoder_lr"], 'weight_decay':Ps["decoder_wd"]}],)
    Ps["optimizer"] = f"{op.__repr__()}\n"
    return op

class Scheduler:

    def __init__(self, optimizer, function_list):

        nGroup = len(optimizer.param_groups)
        self.nGroup = nGroup
        assert nGroup == len(function_list), "bad length."


        self.optimizer = optimizer
        self.function_list = function_list

    def step(self, epoch, argument_list):

        nGroup = self.nGroup
        assert nGroup == len(argument_list), "bad length."
        for item in argument_list:
            assert isinstance(item, tuple), "argument_list should be a list of tuple."

        for param_group, func, arg in zip(self.optimizer.param_groups, self.function_list, argument_list):
            if func is not None:
                param_group["lr"] = func(epoch, *arg)

    def ScheduleFunc1(self, epoch, init_lr):
        return init_lr

    def ScheduleFunc2(self, epoch, init_lr):
        return init_lr

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
