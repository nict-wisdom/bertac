#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import logging
logger = logging.getLogger(__name__)

class BilinearAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:

    * o_i = x_i'Wy for x_i in X.

    """

    def __init__(self, x_size, y_size):
        super(BilinearAttn, self).__init__()
        self.linear = nn.Linear(y_size, x_size)

    def forward(self, x, y):
        """
        Args:
            x: batch * len * hdim1
            y: batch * hdim2
        Output:
            alpha = batch * len
        """
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        alpha = F.softmax(xWy,1)
        alpha = alpha.unsqueeze(2) * x
        return alpha
