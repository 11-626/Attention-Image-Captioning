#-----------------------------------------------------------------------------
# Train and Validate functions and etc.
#-----------------------------------------------------------------------------

import torch, torchvision
from Optimizer import clip_gradient
from Utils import accuracy


def train(Ps, dataloader, encoder, decoder, criterion, optimizer):
    r"""
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param optimizer: optimizer to update encoder and decoder 's weights (if fine-tuning)
    """
    encoder.train()
    decoder.train()

    loss_total = 0.0
    acc_total = 0; acc_ref = 0
    for i, (imgs, caps, caplens) in enumerate(dataloader, 1):
        # Forward prop
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_idx = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores, _ = torch.nn.utils.rnn.pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = torch.nn.utils.rnn.pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += Ps["alpha_c"] * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop
        optimizer.zero_grad()
        loss.backward()

        #Clip gradients
        if Ps["grad_clip"] is not None:
            clip_gradient(optimizer, Ps["grad_clip"])

        # Update weights
        optimizer.step()

        # accumulate accuracy
        top5 = accuracy(scores, targets, 5)
        acc_total += top5[0]; acc_ref += top5[1]

        # accumulate loss
        loss_total += loss.item()

    loss_final = loss_total / i
    acc_final = acc_total / acc_ref * 100.

    return loss_final, acc_final

def validate(Ps, dataloader, encoder, decoder, criterion):
    r"""
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    """

    encoder.eval()
    decoder.eval()

    loss_total = 0.0
    acc_total = 0; acc_ref = 0
    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(dataloader, 1):
            # Forward prop
            imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_idx = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores, _ = torch.nn.utils.rnn.pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = torch.nn.utils.rnn.pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += Ps["alpha_c"] * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # accumulate accuracy
            top5 = accuracy(scores, targets, 5)
            acc_total += top5[0]; acc_ref += top5[1]

            # accumulate loss
            loss_total += loss.item()

    loss_final = loss_total / i
    acc_final = acc_total / acc_ref * 100.

    return loss_final, acc_final
