import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignDraw(nn.Module):
    def __init__(self, T, A, B, dimY, dimLangRNN):
        # batch * seq * size
        self.T = self.T
        self.A = self.A
        self.B = self.B
        self.dimY = dimY
        self.dimLangRNN = dimLangRNN
        
        self.lang = nn.LSTM(input_size=dimY, hidden_size=dimLangRNN, batch_first=True)
        
        
    def forward(self, x):
        image, caption = x
        caption_1hot = F.one_hot(caption, num_classes=self.dimY)
        caption_reverse_1hot = F.one_hot(torch.flip(caption, dims=(1,)), num_classes=self.dimY)
        h_t_lang = torch.cat((self.lang(caption_1hot), self.lang(caption_reverse_1hot)), dim=(2,))
        