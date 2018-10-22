#-----------------------------------------------------------------------------
# Training script
#-----------------------------------------------------------------------------

import torch, torchvision, tensorboardX
import argparse, sys, time, os, json

from Model import Encoder, DecoderWithAttention as Decoder
from Utils import *
from Optimizer import get_optimizer, Scheduler, clip_gradient
from DataLoader import get_dataloaders, Vocabulary
from Train import train, validate

#from nltk.translate.bleu_score import corpus_bleu



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #-- Data parameters
    parser.add_argument("--data_folder", type=str, help="folder with data files saved by create_input_files.py", default="../data/attention_image_caption")
    parser.add_argument("--dataset", type=str, help="which dataset ?", default="flickr8k")
    #-- Model parameters
    parser.add_argument("--embed_dim", type=int, help="dimension of word embeddings", default=512)
    parser.add_argument("--attention_dim", type=int, help="dimension of attention linear layers", default=512)
    parser.add_argument("--decoder_dim", type=int, help="dimension of decoder RNN", default=512)
    parser.add_argument("--dropout", type=float, help="dropout between RNN layer and prediction Linear layer", default=0.5)
    parser.add_argument("--device", type=str, help="cuda or cpu", default=None)
    #-- Training parameters
    parser.add_argument("--num_epochs", type=int, help="maximum number of epochs", default=120)
    parser.add_argument("--batch_size", type=int, help="maximum number of epochs", default=64)
    parser.add_argument("--num_workers", type=int, help="number of cpu to load your data", default=0)
    parser.add_argument("--encoder_lr", type=float, help="learning rate of encoder if fine_tune is True", default=1e-4)
    parser.add_argument("--encoder_wd", type=float, help="weight decay of encoder if fine_tune is True", default=0)
    parser.add_argument("--decoder_lr", type=float, help="learning rate of decoder", default=4e-4)
    parser.add_argument("--decoder_wd", type=float, help="weight decay of decoder", default=0)
    parser.add_argument("--grad_clip", type=float, help="clip gradients at an absolute value", default=5)
    parser.add_argument("--alpha_c", type=float, help="regularization parameter for 'doubly stochastic attention', as in the paper", default=1)
    parser.add_argument("--fine_tune_encoder", type=str, help="whether to fine tune the encoder", default="no")
    parser.add_argument("--limit", type=str, help="whether to limit size of dataset for the sake of dubugging", default="yes")
    parser.add_argument("--seed", type=int, help="seed of random generator", default=1234)
    #-- other parameters
    parser.add_argument("--hist_interval", type=int, help="interval of plotting histogram", default=10)
    parser.add_argument("--save_interval", type=int, help="interval of checkpoint", default=30)
    parser.add_argument("--version", type=str, help="version of this experiment.", default="v1")
    parser.add_argument("--parent", type=str, help="parent of transfer learning.", default=None)

    args = parser.parse_args()
    assert args.fine_tune_encoder in ("yes", "no"), "wrong --fine_tune_encoder argument."
    assert args.limit in ("yes", "no"), "wrong --limit argument."
    assert args.version is not None, "wrong --version argument."

    #-------------------------------------------------------------------------
    #
    #-------------------------------------------------------------------------
    Ps = args2Ps(args)
    #-------------------------------------------------------------------------
    # Vocabulary
    #-------------------------------------------------------------------------
    vocab = Vocabulary()
    vocab.make(dataset="flickr8k", min_word_freq=5)

    #-------------------------------------------------------------------------
    # models
    #-------------------------------------------------------------------------
    encoder = Encoder()
    encoder.fine_tune(Ps["fine_tune_encoder"])
    decoder = Decoder(attention_dim   = Ps["attention_dim"],
                      embed_dim       = Ps["embed_dim"],
                      decoder_dim     = Ps["decoder_dim"],
                      encoder_dim     = encoder.encoder_dim,
                      vocab_size      = len(vocab),
                      device          = Ps["device"],
                      dropout         = Ps["dropout"] )
    encoder = encoder.to(Ps["device"])
    decoder = decoder.to(Ps["device"])
    # whether to load a saved state_dict from checkpoint file
    if Ps["parent"] is not None:
        pass

    #-------------------------------------------------------------------------
    # optimizer and scheduler
    #-------------------------------------------------------------------------
    optimizer = get_optimizer(Ps, encoder, decoder)
    scheduler = Scheduler(optimizer, [None,None])
    #-------------------------------------------------------------------------
    # criterion
    #-------------------------------------------------------------------------
    criterion = torch.nn.CrossEntropyLoss().to(Ps["device"])

    #-------------------------------------------------------------------------
    # custom dataloaders
    #-------------------------------------------------------------------------
    loaders = get_dataloaders(Ps, vocab)

    #-------------------------------------------------------------------------
    # writer
    #-------------------------------------------------------------------------
    writer = make_log(Ps)

    # onnx failed so we comment out this line
    #write_tensorboard_models(writer, encoder, decoder, loaders["valid"])

    try:
        acc_best = 90
        for epoch in range(1, Ps["num_epochs"]+1):
            scheduler.step(epoch, [(None,),(None,)])

            loss_train, acc_train = train(Ps, loaders["train"], encoder, decoder, criterion, optimizer)
            loss_valid, acc_valid = validate(Ps, loaders["valid"], encoder, decoder, criterion)
            get_status(Ps, epoch, loss_train, acc_train, loss_valid, acc_valid)

            #-- write tensorboard
            write_tensorboard_scalar(epoch, writer, type="Loss", Train=loss_train, Valid=loss_valid, Valid_Train=loss_valid-loss_train)
            write_tensorboard_scalar(epoch, writer, type="Accuracy", Train=acc_train, Valid=acc_valid)
            write_tensorboard_scalar(epoch, writer, type="LearningRate", Encoder=optimizer.param_groups[0]['lr'], Decoder=optimizer.param_groups[1]['lr'])
            write_tensorboard_scalar(epoch, writer, type="WeightDecay", Encoder=optimizer.param_groups[0]['weight_decay'], Decoder=optimizer.param_groups[1]['weight_decay'])

            if epoch % Ps["hist_interval"] == 0:
                write_tensorboard_histogram(epoch, writer, encoder, decoder)

            #-- checkpoint
            if Ps["checkpoint"] and acc_valid > acc_best:
                acc_best = acc_valid
                checkpoint("best", Ps, epoch, encoder, decoder, loss_train, loss_valid, acc_train, acc_valid)

            if Ps["checkpoint"] and epoch % Ps["save_interval"] == 0:
                checkpoint("epoch", Ps, epoch, encoder, decoder, loss_train, loss_valid, acc_train, acc_valid)

    except KeyboardInterrupt:
        #-- checkpoint
        if Ps["checkpoint"]:
            checkpoint("interupt", Ps, epoch, encoder, decoder, loss_train, loss_valid, acc_train, acc_valid)
