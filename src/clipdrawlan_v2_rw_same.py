import math
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

import utils

# dimY, dimLangRNN: caption encoding
# dimRNNEnc: LSTM input dimension
# dimZ: latent image dimension
# dimRNNDec: LSTM output dimension
# dimAlign: alignment dimension
class AlignDrawClipLanv2(nn.Module):
    def __init__(
        self,
        args,
        device,
    ):
        super().__init__()
        self.device = device
        self.channels = args.channels

        self.T = args.runSteps
        self.A = args.dimA
        self.B = args.dimB
        self.N = args.dimReadAttent
        self.dimY = args.dimY
        self.dimRNNEnc = args.dimRNNEnc
        self.dimZ = args.dimZ
        self.dimRNNDec = args.dimRNNDec

        self.lang = nn.LSTM(args.dimY, args.dimLangRNN)
        self.clip, preprocess = clip.load('ViT-B/32', device)
        self.lang_map = nn.Sequential(
                            nn.Linear(4*args.dimLangRNN, 2*args.dimLangRNN, bias=True),
                            nn.ReLU()
                            )
        self.encoder = nn.LSTMCell(2 * self.N * self.N * args.channels + args.dimRNNDec, args.dimRNNEnc)
        # Q(z|x) parameters
        self.w_mu = nn.Linear(args.dimRNNEnc, args.dimZ)
        self.w_logvar = nn.Linear(args.dimRNNEnc, args.dimZ)
        # prior P(z) paramters
        self.w_mu_prior = nn.Linear(args.dimRNNDec, args.dimZ)
        self.w_logvar_prior = nn.Linear(args.dimRNNDec, args.dimZ)

        # align parameters
        self.w_lang_align = nn.Linear(2 * args.dimLangRNN, args.dimAlign)
        self.w_dec_align = nn.Linear(args.dimRNNDec, args.dimAlign)
        self.w_v_align = nn.Linear(args.dimAlign, 1)

        self.decoder = nn.LSTMCell(args.dimZ + 2 * args.dimLangRNN, args.dimRNNDec)
        self.w_dec_attn = nn.Linear(args.dimRNNDec, 5)
        self.w_dec = nn.Linear(args.dimRNNDec, self.N * self.N * args.channels)

        # records of canvas matrices, and mu, logvars (used to 0compute loss)
        self.cs = [0] * self.T
        self.mus, self.logvars = [0] * self.T, [0] * self.T
        self.mus_prior, self.logvars_prior = [0] * self.T, [0] * self.T

    def text_encoder(self, caption):
        # seq * B * dimY
        caption_1hot = F.one_hot(caption.to(torch.int64), num_classes=self.dimY).transpose(0, 1).to(torch.float32)
    
        caption_reverse_1hot = (
            F.one_hot(torch.flip(caption.to(torch.int64), dims=(1,)), num_classes=self.dimY)
            .transpose(0, 1)
            .to(torch.float32)
        )
        # output, (hn, cn) = self.lang(caption_1hot)
        # seq * B * (2*dimLangRNN)
        h_t_lang = torch.cat(
            (self.lang(caption_1hot)[0], self.lang(caption_reverse_1hot)[0]), 2
        )

        with torch.no_grad():
            clip_feat = self.clip.encode_text(caption).to(torch.float32)
        n, b, d = h_t_lang.shape
        clip_feat = clip_feat.unsqueeze(0).repeat(n, 1, 1)
        fused_feat = torch.cat([h_t_lang, clip_feat], dim=-1)
        fused_feat = self.lang_map(fused_feat.flatten(0, 1))
        return fused_feat.view(n, b, d)
    
    # def text_encoder(self, caption):
    #     # seq * B * dimY
    #     with torch.no_grad():
    #         text_features = self.clip.encode_text(caption)
    #     return text_features.to(torch.float32)


    def alignment(self, h_dec_prev, h_t_lang):
        ##### compute alignments

        # seq * B * dimAlign
        hid_align = self.w_lang_align(h_t_lang)
        # B * dimAlign
        h_dec_align = self.w_dec_align(h_dec_prev)
        # seq * B * dimAlign
        all_align = torch.tanh(hid_align + h_dec_align)
        # seq * B
        e = self.w_v_align(all_align).squeeze(-1)
        alpha = torch.softmax(e, dim=0)
        # B * (2*dimLangRNN)
        s_t = (alpha[:, :, None] * h_t_lang).sum(axis=0)

        return s_t

    def forward(self, x):
        img, caption = x
        batch_size = caption.size()[0]
        self.batch_size = batch_size

        # lstm encoder parts
        h_enc_prev = torch.zeros(batch_size, self.dimRNNEnc, device=self.device, requires_grad=True)
        h_dec_prev = torch.zeros(batch_size, self.dimRNNDec, device=self.device, requires_grad=True)
        enc_state = torch.zeros(batch_size, self.dimRNNEnc, device=self.device, requires_grad=True)
        dec_state = torch.zeros(batch_size, self.dimRNNDec, device=self.device, requires_grad=True)

        mu_prior = torch.zeros(batch_size, self.dimZ, device=self.device, requires_grad=True)
        logvar_prior = torch.zeros(batch_size, self.dimZ, device=self.device, requires_grad=True)

        lang_feat = self.text_encoder(caption)

        for t in range(self.T):
            c_prev = (
                self.cs[t - 1]
                if t > 0
                else torch.zeros(batch_size, self.A * self.B * self.channels, device=self.device, requires_grad=True)
            )
            # eq. 10 -> 13
            img_hat = img - torch.sigmoid(c_prev)
            r_t = self.read(img, img_hat, h_dec_prev)
            h_enc, enc_state = self.encoder(
                torch.cat((r_t, h_dec_prev), 1), (h_enc_prev, enc_state)
            )
            # Q(z|x)
            mu = self.w_mu(h_enc)
            logvar = self.w_logvar(h_enc)

            z_t = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            self.mus[t], self.logvars[t] = mu, logvar

            # eq. 3 -> 4
            s_t = self.alignment(h_dec_prev, lang_feat)
            h_dec, dec_state = self.decoder(
                torch.cat((z_t, s_t), 1), (h_dec_prev, dec_state)
            )
            self.cs[t] = c_prev + self.write(h_dec)

            # P(z)
            self.mus_prior[t], self.logvars_prior[t] = mu_prior, logvar_prior
            mu_prior = torch.tanh(self.w_mu_prior(h_dec))
            logvar_prior = torch.tanh(self.w_logvar_prior(h_dec))

            h_enc_prev = h_enc
            h_dec_prev = h_dec

    def loss(self, x, myloss=False):
        img, caption = x
        self.forward(x)

        criterion = nn.BCELoss()
        x_recon = torch.sigmoid(self.cs[-1])
        # reconstruction loss
        Lx = criterion(x_recon, img) * self.A * self.B * self.channels
        # kl loss
        Lz = 0  # technically it should be a tensor with Size([batch_size])
        for t in range(self.T):
            # kl loss in DRAW (assuming P(z) ~ N(0, I))
            # kl = 0.5*(torch.sum(self.mus[t]**2 + torch.exp(self.logvars[t]) - self.logvars[t], dim=1) - self.T)

            # The implementation of the official repo seems to be the following equation. I think it's wrong.
            # for each t, kl loss = ((mu - mu_prior)^2 + var) / var_prior - (logvar - logvar_prior) - 1
            if not myloss:
                kl = 0.5 * (
                    torch.sum(
                        ((self.mus[t] - self.mus_prior[t]) ** 2 + torch.exp(self.logvars[t]))
                        / torch.exp(self.logvars_prior[t])
                        - (self.logvars[t] - self.logvars_prior[t]),
                        dim=1,
                    )
                    - self.T
                )
            else:
            # I think the following is correct.
            # for each t, kl loss = (mu - mu_prior)^2 + var / var_prior - (logvar - logvar_prior) - 1
            # This can be derived from computing E_z~q [ln q(z|x) - ln p(z)]

            # reminder: in vae, the objective is to min_{theta, phi} E_z~q [ln q_phi(z|x) - ln p_theta(z)   - ln p_theta(x|z)]
            # the former 2 is called kl loss, and the last is reconstruction loss
                kl = 0.5 * (
                    torch.sum(
                        (self.mus[t] - self.mus_prior[t]) ** 2
                        + torch.exp(self.logvars[t] - self.logvars_prior[t])
                        - (self.logvars[t] - self.logvars_prior[t]), dim=1
                    )
                    - self.T
                )
            Lz += kl

        Lz = torch.mean(Lz)

        return Lx, Lz

    def generate(self, caption):
        batch_size = caption.size(0)
        self.batch_size = batch_size
        h_dec_prev = torch.zeros(batch_size, self.dimRNNDec, device=self.device)
        dec_state = torch.zeros(batch_size, self.dimRNNDec, device=self.device)

        mu_prior = torch.zeros(batch_size, self.dimZ, device=self.device)
        logvar_prior = torch.zeros(batch_size, self.dimZ, device=self.device)

        lang_feat = self.text_encoder(caption)

        for t in range(self.T):
            c_prev = (
                self.cs[t - 1]
                if t > 0
                else torch.zeros(batch_size, self.A * self.B * self.channels, device=self.device)
            )
            z_t = mu_prior + torch.exp(0.5 * logvar_prior) * torch.randn_like(mu_prior)
            s_t = self.alignment(h_dec_prev, lang_feat)
            h_dec, dec_state = self.decoder(
                torch.cat((z_t, s_t), 1), (h_dec_prev, dec_state)
            )
            self.cs[t] = c_prev + self.write(h_dec)
            
            mu_prior = torch.tanh(self.w_mu_prior(h_dec))
            logvar_prior = torch.tanh(self.w_logvar_prior(h_dec))
            h_dec_prev = h_dec

        imgs = []
        for img in self.cs:
            img = torch.sigmoid(img.detach().cpu().view(-1, self.channels, self.A, self.B))
            imgs.append(img)

        return imgs
    
    ######## write
    def write(self, h_dec=0):
        w = self.w_dec(h_dec)
        if self.channels == 3:
            w = w.view(self.batch_size, 3, self.N, self.N)
        elif self.channels == 1:
            w = w.view(self.batch_size, self.N, self.N)
        (Fx, Fy), gamma = self.attn_window(h_dec)
        Fyt = Fy.transpose(self.channels, 2)

        wr = torch.matmul(Fyt, torch.matmul(w, Fx))
        wr = wr.view(self.batch_size, self.A * self.B * self.channels)

        return wr / gamma.view(-1, 1).expand_as(wr)

    ######## read
    def read(self, x, x_hat, h_dec_prev):
        (Fx, Fy), gamma = self.attn_window(h_dec_prev)

        def filter_img(img, Fx, Fy, gamma, A, B, channels, N):
            Fxt = Fx.transpose(channels, 2)
            if channels == 3:
                img = img.view(-1, 3, B, A)
            elif channels == 1:
                img = img.view(-1, B, A)
            glimpse = torch.matmul(Fy, torch.matmul(img, Fxt))
            glimpse = glimpse.view(-1, N * N * channels)

            return glimpse * gamma.view(-1, 1).expand_as(glimpse)

        x = filter_img(x, Fx, Fy, gamma, self.A, self.B, self.channels, self.N)
        x_hat = filter_img(x_hat, Fx, Fy, gamma, self.A, self.B, self.channels, self.N)

        return torch.cat((x, x_hat), 1)

    ########## attention
    def attn_window(self, h_dec):
        params = self.w_dec_attn(h_dec)
        gx_, gy_, log_sigma_2, log_delta, log_gamma = params.split(1, 1)  # 21

        gx = (self.A + 1) / 2 * (gx_ + 1)  # 22
        gy = (self.B + 1) / 2 * (gy_ + 1)  # 23
        delta = (max(self.A, self.B) - 1) / (self.N - 1) * torch.exp(log_delta)  # 24
        sigma2 = torch.exp(log_sigma_2)
        gamma = torch.exp(log_gamma)

        return self.filterbank(gx, gy, sigma2, delta), gamma

    def filterbank(self, gx, gy, sigma2, delta, epsilon=1e-9):
        grid_i = torch.arange(0.0, self.N, device=self.device, requires_grad=True).view(1, -1)

        # Equation 19.
        mu_x = gx + (grid_i - self.N / 2 - 0.5) * delta
        # Equation 20.
        mu_y = gy + (grid_i - self.N / 2 - 0.5) * delta

        a = torch.arange(0.0, self.A, device=self.device, requires_grad=True).view(1, 1, -1)
        b = torch.arange(0.0, self.B, device=self.device, requires_grad=True).view(1, 1, -1)

        mu_x = mu_x.view(-1, self.N, 1)
        mu_y = mu_y.view(-1, self.N, 1)
        sigma2 = sigma2.view(-1, 1, 1)

        Fx = torch.exp(-torch.pow(a - mu_x, 2) / (2 * sigma2))
        Fy = torch.exp(-torch.pow(b - mu_y, 2) / (2 * sigma2))

        Fx = Fx / (Fx.sum(2, True).expand_as(Fx) + epsilon)
        Fy = Fy / (Fy.sum(2, True).expand_as(Fy) + epsilon)

        if self.channels == 3:
            Fx = Fx.view(Fx.size(0), 1, Fx.size(1), Fx.size(2))
            Fx = Fx.repeat(1, 3, 1, 1)

            Fy = Fy.view(Fy.size(0), 1, Fy.size(1), Fy.size(2))
            Fy = Fy.repeat(1, 3, 1, 1)

        return Fx, Fy