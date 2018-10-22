#-----------------------------------------------------------------------------
# model definition
#-----------------------------------------------------------------------------

import torch, torchvision

class Encoder(torch.nn.Module):

    def __init__(self, encoded_image_size=14):
        r"""
        :param encoded_image_size: output feature size of encoded images, optional
        """
        super(Encoder, self).__init__()
        self.encoded_image_size = encoded_image_size

        resnet = torchvision.models.resnet18(pretrained=True) # pretrained cnn model
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.cnn = torch.nn.Sequential(*modules)
        self.encoder_dim = modules[-1][-1].bn2.num_features # if it is not resnet18, maybe `bn3` instead of `bn2`.

        # replace the last pooling layer with a n Adaptive one to get a fixed output size
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((encoded_image_size,encoded_image_size))

    def forward(self, images):
        r"""
        compute a_{i} in section 3.1 (https://arxiv.org/abs/1502.03044)

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.cnn(images)    # (batch_size, 512, image_size/32, image_size/32)
        out = self.adaptive_pool(out) # (batch_size, 512, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1) # (batch_size, encoded_image_size, encoded_image_size, 512)
        #out = out.view(out.size(0), -1, out.size(-1))
        return out


    def fine_tune(self, fine_tune=False):
        r"""
        Allow or prevent the computation of gradients
        for the last 2 Transition and Denseblock, (also the last BatchNorm2d)

        :param fine_tune: Allow?
        """
        for p in self.cnn.parameters():
            p.requires_grad = False
        # if fine_tune is True, compute the gradients for those block we need.
        if fine_tune:
            for m in list(self.cnn.children())[-5:]:
                for p in m.parameters():
                    p.requires_grad = True

class Attention(torch.nn.Module):

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        r"""
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: (hidden) size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()

        #self.encoder_dim = encoder_dim
        #self.attention_dim = attention_dim
        #self.decoder_dim = decoder_dim

        self.encoder_att = torch.nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = torch.nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = torch.nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        r"""
        apply equation (4) and (5) in section 3.1.2 (https://arxiv.org/abs/1502.03044)
        to compute \alpha_{ti} and z_{t}

        "an attention model f_{att} for which we use a multilayer perceptron
        conditioned on the previous hidden state h_{t-1}"

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """

        # torch.nn.Linear's output : "all but the last dimension are the same shape as the input."
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att) # (batch_size, num_pixels)
        # a full image averaged weighted attention map -> sum over pixels : z_{t}
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha

class DecoderWithAttention(torch.nn.Module):

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim, device, dropout=0.5):
        r"""
        :param attention_dim: size of attention network
        :param embed_dim: embedding size (input size of decoder's RNN)
        :param decoder_dim: (hidden) size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.device = device
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = torch.nn.Dropout(p=self.dropout)
        self.decode_step = torch.nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        # f_{init,h} (https://arxiv.org/abs/1502.03044)
        self.init_h = torch.nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        # f_{init,c} (https://arxiv.org/abs/1502.03044)
        self.init_c = torch.nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        # ?
        self.f_beta = torch.nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = torch.nn.Sigmoid()
        # L_{o} in equation (7) (https://arxiv.org/abs/1502.03044)
        self.fc = torch.nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary

        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        r"""
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        r"""
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = torch.nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        r"""
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1) # (batch_size, encoder_dim)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)  # (batch_size, decoder_dim)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        r"""
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

         # Sort input data by decreasing lengths
        caption_lengths, sort_idx = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_idx]
        encoded_captions = encoded_captions[sort_idx]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(self.device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]) )  # (batch_size_t, decoder_dim)
            # we can furthermore apply `L_{o}(Ey_{t-1}+L_{h}h_{t}+L_{z}z_{t})` here
            # instead of `(L_{o}L_{h})h_t` only
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_idx
