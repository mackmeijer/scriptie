# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import torch.nn.functional as F

import egg.core as core
from egg.zoo.signal_game.archs import InformedSender, Receiver
from egg.zoo.signal_game.features import ImageNetFeat, ImagenetLoader


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="")
    parser.add_argument( "--tau_s", type=float, default=10.0)
    parser.add_argument( "--game_size", type=int, default=2)
    parser.add_argument("--same", type=int, default=0)
    parser.add_argument("--embedding_size", type=int, default=50)
    parser.add_argument( "--hidden_size",  type=int,  default=20)
    parser.add_argument( "--batches_per_epoch", type=int,  default=100)
    parser.add_argument("--inf_rec", type=int, default=0)
    parser.add_argument( "--mode",  type=str, default="rf")
    parser.add_argument("--gs_tau", type=float, default=1.0)

    opt = core.init(parser)
    assert opt.game_size >= 1

    return opt


def loss(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
    """
    Accuracy loss - non-differetiable hence cannot be used with GS
    """
    acc = (labels == receiver_output).float()
    return -acc, {"acc": acc}


def loss_nll(
    _sender_input, _message, _receiver_input, receiver_output, labels, _aux_input
):
    """
    NLL loss - differentiable and can be used with both GS and Reinforce
    """
    nll = F.nll_loss(receiver_output, labels, reduction="none")
    acc = (labels == receiver_output.argmax(dim=1)).float().mean()
    return nll, {"acc": acc}


def get_game(opt):
    feat_size = 4096
    sender = InformedSender(
        opt.game_size,
        feat_size,
        opt.embedding_size,
        opt.hidden_size,
        opt.vocab_size,
        temp=opt.tau_s,
    )
    receiver = Receiver(
        opt.game_size,
        feat_size,
        opt.embedding_size,
        opt.vocab_size,
        reinforce=(opts.mode == "rf"),
    )
    if opts.mode == "rf":
        sender = core.ReinforceWrapper(sender)
        receiver = core.ReinforceWrapper(receiver)
        game = core.SymbolGameReinforce(
            sender,
            receiver,
            loss,
            sender_entropy_coeff=0.01,
            receiver_entropy_coeff=0.01,
        )
    elif opts.mode == "gs":
        sender = core.GumbelSoftmaxWrapper(sender, temperature=opt.gs_tau)
        game = core.SymbolGameGS(sender, receiver, loss_nll)
    else:
        raise RuntimeError(f"Unknown training mode: {opts.mode}")

    return game



opts = parse_arguments()
game = get_game(opts)
optimizer = core.build_optimizer(game.parameters())
callback = None

if opts.mode == "gs":
    callbacks = [core.TemperatureUpdater(agent=game.sender, decay=0.9, minimum=0.1)]
else:
    callbacks = []

callbacks.append(core.ConsoleLogger(as_json=True, print_train_loss=True))
trainer = core.Trainer(
    game=game,
    optimizer=optimizer,
    train_data= get_batch(100),
    validation_data= get_batch(20),
    callbacks=callbacks,
)

trainer.train(n_epochs=opts.n_epochs)

core.close()