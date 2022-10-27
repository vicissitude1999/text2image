import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

# dimY, dimLangRNN: caption encoding
# dimRNNEnc: LSTM input dimension
# dimZ: latent image dimension
# dimRNNDec: LSTM output dimension
# dimAlign: alignment dimension
class AlignDraw(nn.Module):
    def __init__(
        self,
        runSteps,
        dimReadAttent,  # dimReadAttent = dimWriteAttent
        dimWriteAttent,
        dimX,
        dimY,
        dimLangRNN,
        dimRNNEnc,
        dimZ,
        dimRNNDec,
        dimAlign,
        device,
    ):
        super().__init__()
        self.device = device
        self.T = runSteps
        self.A = int(math.sqrt(dimX))
        self.B = int(math.sqrt(dimX))
        self.N = dimReadAttent
        self.dimY = dimY
        self.dimLangRNN = dimLangRNN
        self.dimRNNEnc = dimRNNEnc
        self.dimZ = dimZ
        self.dimRNNDec = dimRNNDec

        self.lang = nn.LSTM(dimY, dimLangRNN)
        self.encoder = nn.LSTMCell(2 * self.N * self.N + dimRNNDec, dimRNNEnc)
        # Q(z|x) parameters
        self.w_mu = nn.Linear(dimRNNEnc, dimZ)
        self.w_logvar = nn.Linear(dimRNNEnc, dimZ)
        # prior P(z) paramters
        self.w_mu_prior = nn.Linear(dimRNNDec, dimZ)
        self.w_logvar_prior = nn.Linear(dimRNNDec, dimZ)

        # align parameters
        self.w_lang_align = nn.Linear(2 * dimLangRNN, dimAlign)
        self.w_dec_align = nn.Linear(dimRNNDec, dimAlign)
        self.w_v_align = nn.Linear(dimAlign, 1)

        self.decoder = nn.LSTMCell(dimZ + 2 * dimLangRNN, dimRNNDec)
        self.dec_linear = nn.Linear(dimRNNDec, 5)
        self.dec_w_linear = nn.Linear(dimRNNDec, self.N * self.N)

        # records of canvas matrices, and mu, logvars (used to compute loss)
        self.cs = [0] * self.T
        self.mus, self.logvars = [0] * self.T, [0] * self.T
        self.mus_prior, self.logvars_prior = [0] * self.T, [0] * self.T

    def alignment(self, h_dec_prev, caption):
        ##### compute alignments
        # seq * B * dimY
        caption_1hot = (
            F.one_hot(caption, num_classes=self.dimY).transpose(0, 1).to(torch.float32)
        )
        caption_reverse_1hot = (
            F.one_hot(torch.flip(caption, dims=(1,)), num_classes=self.dimY)
            .transpose(0, 1)
            .to(torch.float32)
        )
        # output, (hn, cn) = self.lang(caption_1hot)
        # seq * B * (2*dimLangRNN)
        h_t_lang = torch.cat(
            (self.lang(caption_1hot)[0], self.lang(caption_reverse_1hot)[0]), 2
        )

        # seq * B * dimAlign
        hid_align = self.w_lang_align(h_t_lang)
        # B * dimAlign
        h_dec_align = self.w_dec_align(h_dec_prev)
        # seq * B * dimAlign
        all_align = torch.tanh(hid_align + h_dec_align)
        # seq * B
        e = self.w_v_align(all_align).squeeze()
        alpha = torch.softmax(e, dim=0)
        # B * (2*dimLangRNN)
        s_t = (alpha[:, :, None] * h_t_lang).sum(axis=0)

        return s_t

    def forward(self, x):
        img, caption = x
        batch_size = caption.size()[0]
        self.batch_size = batch_size

        # lstm encoder parts
        h_enc_prev = torch.zeros(batch_size, self.dimRNNEnc, device=self.device)
        h_dec_prev = torch.zeros(batch_size, self.dimRNNDec, device=self.device)
        enc_state = torch.zeros(
            batch_size, self.dimRNNEnc, device=self.device
        )  # cell in lstm
        dec_state = torch.zeros(batch_size, self.dimRNNDec, device=self.device)

        mu_prior = torch.zeros(batch_size, self.dimZ, device=self.device)
        logvar_prior = torch.zeros(batch_size, self.dimZ, device=self.device)

        s_t = self.alignment(h_dec_prev, caption)

        for t in range(self.T):
            c_prev = (
                self.cs[t - 1]
                if t > 0
                else torch.zeros(batch_size, self.A * self.B, device=self.device)
            )

            img_hat = img - torch.sigmoid(c_prev)
            r_t = self.read(img, img_hat, h_dec_prev)
            h_enc, enc_state = self.encoder(
                torch.cat((r_t, h_dec_prev), 1), (h_enc_prev, enc_state)
            )

            mu = self.w_mu(h_enc)
            logvar = self.w_logvar(h_enc)

            # Q(z|x)
            z_t = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            self.mus[t], self.logvars[t] = mu, logvar

            h_dec, dec_state = self.decoder(
                torch.cat((z_t, s_t), 1), (h_dec_prev, dec_state)
            )
            self.cs[t] = c_prev + self.write(h_dec)

            # P(z)
            self.mus_prior[t], self.logvars[t] = mu_prior, logvar_prior
            mu_prior = torch.tanh(self.w_mu_prior(h_dec))
            logvar_prior = torch.tanh(self.w_logvar_prior(h_dec))

            h_enc_prev = h_enc
            h_dec_prev = h_dec

    def loss(self, x):
        img, caption = x
        self.forward(x)

        criterion = nn.BCELoss()
        x_recon = torch.sigmoid(self.cs[-1])
        # reconstruction loss
        Lx = criterion(x_recon, img) * self.A * self.B
        # kl loss
        Lz = 0
        for t in range(self.T):
            # kl loss in DRAW (assuming P(z) ~ N(0, I))
            # kl = 0.5*(torch.sum(self.mus[t]**2 + torch.exp(self.logvars[t]) - self.logvars[t]) - self.T)
            
            # The implementation of the official repo seems to be the following equation. I think it's wrong.
            # for each t, kl loss = ((mu - mu_prior)^2 + var) / var_prior - (logvar - logvar_prior) - 1
            # kl = 0.5*(torch.sum(((self.mus[t] - self.mus_prior[t])**2 + torch.exp(self.logvars[t])) /
            #                     torch.exp(self.logvars_prior[t]) -
            #                     (self.logvars[t] - self.logvars_prior[t])) -
            #                     self.T)

            # I think the following is correct.
            # for each t, kl loss = (mu - mu_prior)^2 + var / var_prior - (logvar - logvar_prior) - 1
            # This can be derived from computing E_z~q [ln q(z|x) - ln p(z)]
            
            # reminder: in vae, the objective is to min_{theta, phi} E_z~q [ln q_phi(z|x) - ln p_theta(z)   - ln p_theta(x|z)]
            # the former 2 is called kl loss, and the last is reconstruction loss
            kl = 0.5 * (
                torch.sum(
                    (self.mus[t] - self.mus_prior[t]) ** 2
                    + torch.exp(self.logvars[t] - self.logvars_prior[t])
                    - (self.logvars[t] - self.logvars_prior[t])
                )
                - self.T
            )

            Lz += kl
        Lz = torch.mean(Lz)
        
        return Lx + Lz
    
    def generate(self, caption, batch_size=64):
        # self.batch_size = batchsize
        h_dec_prev = torch.zeros(batch_size, self.dimRNNDec, device=self.device)
        dec_state = torch.zeros(batch_size, self.dimRNNDec, device=self.device)

        mu_prior = torch.zeros(batch_size, self.dimZ, device=self.device)
        logvar_prior = torch.zeros(batch_size, self.dimZ, device=self.device)

        s_t = self.alignment(h_dec_prev, caption)

        for t in range(self.T):
            c_prev = (
                self.cs[t - 1]
                if t > 0
                else torch.zeros(batch_size, self.A, self.B, device=self.device)
            )
            z_t = mu_prior + torch.exp(1 / 2 * logvar_prior) * torch.randn_like(
                mu_prior
            )
            h_dec, dec_state = self.decoder(
                torch.cat((z_t, s_t), 1), (h_dec_prev, dec_state)
            )
            self.cs[t] = c_prev + self.write(h_dec)

            h_dec_prev = h_dec

        imgs = []
        for img in self.cs:
            img = img.view(-1, self.B, self.A)
            imgs.append(torch.sigmoid(img).detach().cpu().numpy())

        return imgs

    ######## write
    def write(self, h_dec=0):
        w = self.dec_w_linear(h_dec)
        w = w.view(self.batch_size, self.N, self.N)
        # w = Variable(torch.ones(4,5,5) * 3)
        # self.batch_size = 4
        (Fx, Fy), gamma = self.attn_window(h_dec)
        Fyt = Fy.transpose(2, 1)
        # wr = matmul(Fyt,matmul(w,Fx))
        wr = Fyt.bmm(w.bmm(Fx))
        wr = wr.view(self.batch_size, self.A * self.B)

        return wr / gamma.view(-1, 1).expand_as(wr)

    ######## read
    def read(self, x, x_hat, h_dec_prev):
        (Fx, Fy), gamma = self.attn_window(h_dec_prev)

        def filter_img(img, Fx, Fy, gamma, A, B, N):
            Fxt = Fx.transpose(2, 1)
            img = img.view(-1, B, A)
            # img = img.transpose(2,1)
            # glimpse = matmul(Fy,matmul(img,Fxt))P
            glimpse = Fy.bmm(img.bmm(Fxt))
            glimpse = glimpse.view(-1, N * N)
            return glimpse * gamma.view(-1, 1).expand_as(glimpse)

        x = filter_img(x, Fx, Fy, gamma, self.A, self.B, self.N)
        x_hat = filter_img(x_hat, Fx, Fy, gamma, self.A, self.B, self.N)

        return torch.cat((x, x_hat), 1)

    ########## attention
    def attn_window(self, h_dec):
        params = self.dec_linear(h_dec)
        gx_, gy_, log_sigma_2, log_delta, log_gamma = params.split(1, 1)  # 21

        gx = (self.A + 1) / 2 * (gx_ + 1)  # 22
        gy = (self.B + 1) / 2 * (gy_ + 1)  # 23
        delta = (max(self.A, self.B) - 1) / (self.N - 1) * torch.exp(log_delta)  # 24
        sigma2 = torch.exp(log_sigma_2)
        gamma = torch.exp(log_gamma)

        return self.filterbank(gx, gy, sigma2, delta), gamma

    def filterbank(self, gx, gy, sigma2, delta):
        rng = torch.arange(0, self.N, device=self.device).view(1, -1)
        mu_x = self.compute_mu(gx, rng, delta)
        mu_y = self.compute_mu(gy, rng, delta)

        a = torch.arange(0, self.A, device=self.device).view(1, 1, -1)
        b = torch.arange(0, self.B, device=self.device).view(1, 1, -1)

        mu_x = mu_x.view(-1, self.N, 1)
        mu_y = mu_y.view(-1, self.N, 1)
        sigma2 = sigma2.view(-1, 1, 1)

        Fx = self.filterbank_matrices(a, mu_x, sigma2)
        Fy = self.filterbank_matrices(b, mu_y, sigma2)

        return Fx, Fy

    def filterbank_matrices(self, a, mu_x, sigma2, epsilon=1e-9):
        t_a, t_mu_x = utils.align(a, mu_x)
        temp = t_a - t_mu_x
        temp, t_sigma = utils.align(temp, sigma2)
        temp = temp / (t_sigma * 2)
        F = torch.exp(-torch.pow(temp, 2))
        F = F / (F.sum(2, True).expand_as(F) + epsilon)
        return F

    def compute_mu(self, g, rng, delta):
        rng_t, delta_t = utils.align(rng, delta)
        tmp = (rng_t - self.N / 2 - 0.5) * delta_t
        tmp_t, g_t = utils.align(tmp, g)
        mu = tmp_t + g_t
        return mu
