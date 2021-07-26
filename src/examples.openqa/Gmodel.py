#!/usr/bin/env python3
# Copyright (c) 2021-present, Data-driven Intelligent System Research Center (DIRECT), National Institute of Information and Communications Technology (NICT).
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import cnn_utils

class Generator(nn.Module):

    def __init__(self, args):
        super(Generator,self).__init__()
        self.args = args
        self.eps = 1e-8
        self.emb_params = cnn_utils.get_emb_params(args)

        self.embs = nn.ModuleList([nn.Embedding(size, dim, padding_idx=pad)
                                   for size, dim, pad in self.emb_params])
        self.emb_dim = args.emb_dim
        self.shift_emb = nn.Linear(args.emb_dim, args.emb_dim, False)
        nn.init.normal_(self.shift_emb.weight, 0, 0.1) 

        self.conv = nn.ModuleList()
        for w in args.filter_widths:
            conv0 = nn.Conv2d(1, args.filter_size, (w, self.emb_dim),
                                      padding=(w - 1, 0))
            if args.cnn_init == 'xavier':
              nn.init.xavier_uniform_(conv0.weight)
            elif args.cnn_init == 'he':
              nn.init.kaiming_uniform_(conv0.weight)
            else:
              nn.init.uniform_(conv0.weight, -0.01, 0.01)
              nn.init.constant_(conv0.bias, 0.0) 
            self.conv.append(conv0)
        self.fdim = len(args.filter_widths) * args.filter_size

        self.init_parameters()

    def init_parameters(self):
        for emb, (size, dim, pad) in list(zip(self.embs, self.emb_params)):
            emb.weight.data.uniform_(-0.1, 0.1)
            emb.weight.data[pad].zero_()

    def forward(self, v_cols, hq=None, mode=0):
        out = []
        o_out = []
        if mode == 0:
            q_emb = self.embs[0](v_cols[0])
            return q_emb, self.fdim
        else:
            out.append(F.leaky_relu(self.shift_emb(hq)))
        qout = [F.leaky_relu(conv(out[0].unsqueeze(1)))\
               .squeeze(3) for i, conv in enumerate(self.conv)]
        qout = torch.cat([F.avg_pool1d(o, o.size(2)).squeeze(2) for o in qout],1 )

        return qout, self.fdim
