#-----------------------------------------------------------------------#
#                          Library imports                              #
#-----------------------------------------------------------------------#
import os
import torch


#-----------------------------------------------------------------------#
#                             select_device                             #
#     Picks the best available torch device for training/inference      #
#-----------------------------------------------------------------------#
# The original code was written for Google Colab and called .cuda()     #
# unconditionally. This module keeps that behaviour on a CUDA machine   #
# while allowing the same code to run on Apple Silicon (mps) or CPU.    #
#                                                                       #
# Set the SDD_DEVICE environment variable to force a device, e.g.       #
#   SDD_DEVICE=cpu python main.py                                       #
#-----------------------------------------------------------------------#
def select_device():
    forced = os.environ.get('SDD_DEVICE')
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


DEVICE = select_device()
