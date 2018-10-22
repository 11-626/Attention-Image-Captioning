#-----------------------------------------------------------------------------
# utility function and class
#-----------------------------------------------------------------------------

import os, json
import torch, tensorboardX

#-----------------------------------------------------------------------------
# create a Parameter dictionary from argparse
#-----------------------------------------------------------------------------
def args2Ps(args):

    Ps = {
        "data_folder"   : args.data_folder,
        "dataset"       : args.dataset,

        "embed_dim"     : args.embed_dim,
        "attention_dim" : args.attention_dim,
        "decoder_dim"   : args.decoder_dim,
        "dropout"       : args.dropout,

        "num_epochs"    : args.num_epochs,
        "batch_size"    : args.batch_size,
        "num_workers"   : args.num_workers,
        "encoder_lr"    : args.encoder_lr,
        "encoder_wd"    : args.encoder_wd,
        "decoder_lr"    : args.decoder_lr,
        "decoder_wd"    : args.decoder_wd,
        "grad_clip"     : args.grad_clip,
        "alpha_c"       : args.alpha_c,
        "fine_tune_encoder" : True if args.fine_tune_encoder=="yes" else False,
        "isLimit"       : True if args.limit == "yes" else False,
        "seed"          : args.seed,
        "device"        : torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device is None else torch.device(args.device),

        "hist_interval" : args.hist_interval,
        "save_interval" : args.save_interval,
        "version"       : args.version,
        "parent"        : args.parent,
        "checkpoint"    : False, #False if args.limit == "yes" else True,
    }


    return Ps

#-----------------------------------------------------------------------------
# compute accuracy
#-----------------------------------------------------------------------------

def accuracy(scores, targets, k):
    r"""
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, idx = scores.topk(k, 1, True, True)
    correct = idx.eq(targets.view(-1,1).expand_as(idx))
    correct_total = correct.view(-1).float().sum() # 0D tensor

    return correct_total.item(), batch_size

#-----------------------------------------------------------------------------
# Log
#-----------------------------------------------------------------------------

def make_log(Ps):

    if not Ps["isLimit"]:
        #-- tensorboardX
        log_folder = f"./Logs/{Ps['version']}"
        model_folder = f"./Models/{Ps['version']}"
        Ps["log_folder"] = log_folder
        Ps["model_folder"] = model_folder
        assert not os.path.exists(log_folder), f"{log_folder} already exist."
        writer = tensorboardX.SummaryWriter(log_folder)
        #-- heads of info.txt
        info = f"attention image caption ({Ps['dataset']})"
        info+= "\nParameters:\n"
        for key in Ps.keys():
            info += f"{key:20s} : {Ps[key]} \n"
        info += "-"*60 + '\n'
        with open(f"{log_folder}/info.txt", 'a+') as file:
            file.write(info)
        return writer
    else :
        return None

def add_log(Ps, string):
    if not Ps["isLimit"]:
        with open(f"{Ps['log_folder']}/info.txt", 'a+') as file:
            file.write(string)

def get_status(Ps, epoch, loss_train, acc_train, loss_valid, acc_valid):
    status = f"Epoch [{epoch:>4d}/{Ps['num_epochs']:<4d}] --> \n"
    status += f"|-- Loss/Train : {loss_train:>2.5f}\tLoss/Valid : {loss_valid:>2.5f}\n"
    status += f"|-- Accu/Train : {acc_train:>2.5f}%\tAccu/Valid : {acc_valid:>2.5f}%\n"

    if not Ps["isLimit"]:
        with open(f"{Ps['log_folder']}/info.txt", 'a+') as file:
            file.write(status)
    print(status, end='')

def write_tensorboard_scalar(epoch, writer, type, **kwargs):
    if writer is not None:
        for key, value in kwargs.items():
            writer.add_scalar(f'{type}/{key}', value, epoch)

def write_tensorboard_models(writer, encoder, decoder, dataloader):
    if writer is not None:
        for i, (imgs, caps, caplens, allcaps) in enumerate(dataloader, 1):
            writer.add_graph(encoder, (imgs,))
            imgs = encoder(imgs)
            writer.add_graph(decoder, (imgs, caps, caplens,))
            break

def write_tensorboard_histogram(epoch, writer, encoder, decoder):

    if writer is not None:
        for name, param in encoder.named_parameters():
            writer.add_histogram(f"Encoder/{name}", param.clone().cpu().data.numpy(), epoch)
        for name, param in decoder.named_parameters():
            writer.add_histogram(f"Dncoder/{name}", param.clone().cpu().data.numpy(), epoch)

#-----------------------------------------------------------------------------
# checkpoint
#-----------------------------------------------------------------------------

def checkpoint(key, Ps, epoch, encoder, decoder, loss_train, loss_valid, acc_train, acc_valid):
    assert key in ("best", "epoch", "interupt"), "bad key argument."
    print(f"!!! checkpoint {key} at epoch: {epoch}")
    if not os.path.exists(f"{Ps['model_folder']}/") : os.mkdir(f"{Ps['model_folder']}/")
    checkpoint = {
                    "encoder" : encoder.state_dict(),
                    "decoder" : decoder.state_dict(),
                    "Loss" : {"train" : loss_train, "valid" : loss_valid},
                    "Accuracy" : {"train" : acc_train, "valid" : acc_valid},
                }
    if key == "best":
        torch.save(checkpoint, os.path.join(f"{Ps['model_folder']}/", f"checkpoint_{key}.pkl") )
    elif key == "epoch":
        torch.save(checkpoint, os.path.join(f"{Ps['model_folder']}/", f"checkpoint_{key}_{epoch}.pkl") )
    elif key == "interupt":
        torch.save(checkpoint, os.path.join(f"{Ps['model_folder']}/", f"checkpoint_{key}_{epoch}.pkl") )
