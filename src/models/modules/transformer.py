# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional, Tuple
from torch import Tensor, nn
from torch.nn import functional as F
from .attention import Attention


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "elu":
        return F.elu
    raise RuntimeError("activation should be relu/gelu/elu, not {}".format(activation))


class TransformerBlock(nn.Module):
    __constants__ = ["norm"]

    def __init__(
        self,
        d_model: int,
        n_head: int = 2,
        d_feedforward: int = 256,
        dropout_p: float = 0.1,
        activation: str = "relu",
        n_layer: int = 1,
        norm_first: bool = True,
        decoder_self_attn: bool = False,
        bias: bool = True,
        out_layernorm: bool = False,
    ) -> None:
        super(TransformerBlock, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerCrossAttention(
                    d_model=d_model,
                    n_head=n_head,
                    d_feedforward=d_feedforward,
                    dropout_p=dropout_p,
                    activation=activation,
                    norm_first=norm_first,
                    decoder_self_attn=decoder_self_attn,
                    bias=bias,
                )
                for _ in range(n_layer)
            ]
        )

        self.out_layernorm = nn.LayerNorm(d_model) if out_layernorm else None

    def forward(
        self,
        src: Tensor,
        src_padding_mask: Optional[Tensor] = None,
        tgt: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        decoder_tgt: Optional[Tensor] = None,
        decoder_tgt_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            src: [n_batch, n_src, d_model]
            src_padding_mask: [n_batch, n_src], bool, if True, src is invalid.
            tgt: [n_batch, (n_src), n_tgt, d_model], None for self attention, (n_src) if using knn.
            tgt_padding_mask: [n_batch, (n_src), n_tgt], bool, if True, tgt is invalid, (n_src) if using knn.
            decoder_tgt: [n_batch, (n_src), n_tgt_decoder, d_model], (n_src) if using decoder_knn.
            decoder_tgt_padding_mask: [n_batch, (n_src), n_tgt_decoder], (n_src) if using decoder_knn.
            attn_mask: [n_batch, n_src, n_tgt], bool, if True, attn is disabled for that pair of src/tgt.

        Returns:
            src: [n_batch, n_src, d_model]
            attn_weights: [n_batch, n_src, n_tgt] if need_weights else None

        Remarks:
            absoulte_pe should be already added to src/tgt.
        """
        attn_weights = None
        for mod in self.layers:
            src, attn_weights = mod(
                src=src,
                src_padding_mask=src_padding_mask,
                tgt=tgt,
                tgt_padding_mask=tgt_padding_mask,
                decoder_tgt=decoder_tgt,
                decoder_tgt_padding_mask=decoder_tgt_padding_mask,
                attn_mask=attn_mask,
                need_weights=need_weights,
            )
        if self.out_layernorm is not None:
            src = self.out_layernorm(src)
        return src, attn_weights


class TransformerCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_feedforward: int,
        dropout_p: float,
        activation: str,
        norm_first: bool,
        decoder_self_attn: bool,
        bias: bool,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.d_feedforward = d_feedforward
        self.decoder_self_attn = decoder_self_attn
        inplace = False

        self.dropout = nn.Dropout(p=dropout_p, inplace=inplace) if dropout_p > 0 else None
        self.activation = _get_activation_fn(activation)
        self.norm1 = nn.LayerNorm(d_model)

        if self.decoder_self_attn:
            self.attn_src = Attention(d_model=d_model, n_head=n_head, dropout_p=dropout_p, bias=bias)
            self.norm_src = nn.LayerNorm(d_model)
            self.dropout_src = nn.Dropout(p=dropout_p, inplace=inplace) if dropout_p > 0 else None

        if self.norm_first:
            self.norm_tgt = nn.LayerNorm(d_model)

        self.attn = Attention(d_model=d_model, n_head=n_head, dropout_p=dropout_p, bias=bias)
        if self.d_feedforward > 0:
            self.linear1 = nn.Linear(d_model, d_feedforward)
            self.linear2 = nn.Linear(d_feedforward, d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(p=dropout_p, inplace=inplace) if dropout_p > 0 else None
            self.dropout2 = nn.Dropout(p=dropout_p, inplace=inplace) if dropout_p > 0 else None

    def forward(
        self,
        src: Tensor,
        src_padding_mask: Optional[Tensor] = None,
        tgt: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        decoder_tgt: Optional[Tensor] = None,
        decoder_tgt_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            src: [n_batch, n_src, d_model]
            src_padding_mask: [n_batch, n_src], bool, if True, src is invalid.
            tgt: [n_batch, (n_src), n_tgt, d_model], None for self attention, (n_src) if using knn.
            tgt_padding_mask: [n_batch, (n_src), n_tgt], bool, if True, tgt is invalid, (n_src) if using knn.
            decoder_tgt: [n_batch, n_src, n_tgt_decoder, d_model], when use decoder_knn
            decoder_tgt_padding_mask: [n_batch, n_src, n_tgt_decoder], when use decoder_knn
            attn_mask: [n_batch, n_src, n_tgt], bool, if True, attn is disabled for that pair of src/tgt.

        Returns:
            out: [n_batch, n_src, d_model]
            attn_weights: [n_batch, n_src, n_tgt] if need_weights else None
        """
        if self.decoder_self_attn:
            # transformer decoder
            if self.norm_first:
                _s = self.norm_src(src)
                if decoder_tgt is None:
                    _s = self.attn_src(_s, tgt_padding_mask=src_padding_mask)[0]
                else:
                    decoder_tgt = self.norm_src(decoder_tgt)
                    _s = self.attn_src(_s, decoder_tgt, tgt_padding_mask=decoder_tgt_padding_mask)[0]

                if self.dropout_src is None:
                    src = src + _s
                else:
                    src = src + self.dropout_src(_s)
            else:
                if decoder_tgt is None:
                    _s = self.attn_src(src, tgt_padding_mask=src_padding_mask)[0]
                else:
                    _s = self.attn_src(src, decoder_tgt, tgt_padding_mask=decoder_tgt_padding_mask)[0]

                if self.dropout_src is None:
                    src = self.norm_src(src + _s)
                else:
                    src = self.norm_src(src + self.dropout_src(_s))

        if tgt is None:
            tgt_padding_mask = src_padding_mask

        if self.norm_first:
            src2 = self.norm1(src)
            if tgt is not None:
                tgt = self.norm_tgt(tgt)
        else:
            src2 = src

        # [n_batch, n_src, d_model]
        src2, attn_weights = self.attn(
            src=src2, tgt=tgt, tgt_padding_mask=tgt_padding_mask, attn_mask=attn_mask, need_weights=need_weights
        )

        if self.d_feedforward > 0:
            if self.dropout1 is None:
                src = src + src2
            else:
                src = src + self.dropout1(src2)

            if self.norm_first:
                src2 = self.norm2(src)
            else:
                src = self.norm1(src)
                src2 = src

            src2 = self.activation(self.linear1(src2))
            if self.dropout is None:
                src2 = self.linear2(src2)
            else:
                src2 = self.linear2(self.dropout(src2))

            if self.dropout2 is None:
                src = src + src2
            else:
                src = src + self.dropout2(src2)

            if not self.norm_first:
                src = self.norm2(src)
        else:
            # densetnt vectornet
            src2 = self.activation(src2)
            if self.dropout is None:
                src = src + src2
            else:
                src = src + self.dropout(src2)
            if not self.norm_first:
                src = self.norm1(src)

        if src_padding_mask is not None:
            src.masked_fill_(src_padding_mask.unsqueeze(-1), 0.0)
            if need_weights:
                attn_weights.masked_fill_(src_padding_mask.unsqueeze(-1), 0.0)
        return src, attn_weights
