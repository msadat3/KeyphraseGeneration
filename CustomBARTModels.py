from transformers import BartForConditionalGeneration
from transformers.modeling_bart import *
import copy

def _prepare_bart_decoder_inputs(
    config, input_ids, decoder_input_ids=None, decoder_padding_mask=None, causal_mask_dtype=torch.float32
):
    """Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if
    none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
    Note: this is not called during generation
    """
    pad_token_id = config.pad_token_id
    if decoder_input_ids is None:
       # print('None?WTF inside func')
        decoder_input_ids = shift_tokens_right(input_ids, pad_token_id)
    bsz, tgt_len = decoder_input_ids.size()
    if decoder_padding_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
    else:
        decoder_padding_mask = invert_mask(decoder_padding_mask)
    causal_mask = torch.triu(fill_with_neg_inf(torch.zeros(tgt_len, tgt_len)), 1).to(
        dtype=causal_mask_dtype, device=decoder_input_ids.device
    )
    return decoder_input_ids, decoder_padding_mask, causal_mask

def _filter_out_falsey_values(tup) -> Tuple:
    """Remove entries that are None or [] from an iterable."""
    return tuple(x for x in tup if isinstance(x, torch.Tensor) or x)

def _make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


def _reorder_buffer(attn_cache, new_order):
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None:
            attn_cache[k] = input_buffer_k.index_select(0, new_order)
    return attn_cache

class BartModelTwoDecoders(PretrainedBartModel):
    def __init__(self, config, shared, encoder, decoder):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = shared

        self.encoder = encoder
        self.decoder_present = decoder
        self.decoder_absent = copy.deepcopy(decoder)

        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            decoder_present_input_ids=None,
            decoder_absent_input_ids=None,
            encoder_outputs: Optional[Tuple] = None,
            decoder_present_attention_mask=None,
            decoder_absent_attention_mask=None,
            decoder_present_cached_states=None,
            decoder_absent_cached_states=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        if decoder_present_input_ids is None or decoder_absent_input_ids is None:
            #print("None?WTF")
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if not use_cache:
            decoder_present_input_ids, decoder_present_padding_mask, causal_mask_present = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_present_input_ids,
                decoder_padding_mask=decoder_present_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
            decoder_absent_input_ids, decoder_absent_padding_mask, causal_mask_absent = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_absent_input_ids,
                decoder_padding_mask=decoder_absent_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_present_padding_mask, causal_mask_present = None, None
            decoder_absent_padding_mask, causal_mask_absent = None, None

        assert (decoder_present_input_ids is not None and decoder_absent_input_ids is not None)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        assert isinstance(encoder_outputs, tuple)
       # print('present',decoder_present_input_ids)
        #print('absent',decoder_absent_input_ids)

        decoder_present_outputs = self.decoder_present(
            decoder_present_input_ids,
            encoder_outputs[0],
            attention_mask,
            decoder_present_padding_mask,
            decoder_causal_mask=causal_mask_present,
            decoder_cached_states=decoder_present_cached_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )
        decoder_absent_outputs = self.decoder_absent(
            decoder_absent_input_ids,
            encoder_outputs[0],
            attention_mask,
            decoder_absent_padding_mask,
            decoder_causal_mask=causal_mask_absent,
            decoder_cached_states=decoder_absent_cached_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )

        # Attention and hidden_states will be [] or None if they aren't needed
        decoder_present_outputs: Tuple = _filter_out_falsey_values(decoder_present_outputs)
        assert isinstance(decoder_present_outputs[0], torch.Tensor)
        decoder_absent_outputs: Tuple = _filter_out_falsey_values(decoder_absent_outputs)
        assert isinstance(decoder_absent_outputs[0], torch.Tensor)
        encoder_outputs: Tuple = _filter_out_falsey_values(encoder_outputs)

        return decoder_present_outputs + decoder_absent_outputs + encoder_outputs

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.shared)  # make it on the fly


class BartForConditionalGenerationTwoDecoders(PretrainedBartModel):
    base_model_prefix = "model"

    def __init__(self, pre_trained_model):
       # pre_trained_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        super().__init__(pre_trained_model.config)
        base_model = BartModelTwoDecoders(pre_trained_model.config, pre_trained_model.model.shared, pre_trained_model.model.encoder, pre_trained_model.model.decoder)
        self.model = base_model
        self.register_buffer("final_logits_bias_present", torch.zeros((1, self.model.shared.num_embeddings)))
        self.register_buffer("final_logits_bias_absent", torch.zeros((1, self.model.shared.num_embeddings)))

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        print('resizing............')
        old_num_tokens = self.model.shared.num_embeddings
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self.model.shared = new_embeddings
        self._resize_final_logits_bias(new_num_tokens, old_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int, old_num_tokens: int) -> None:
        if new_num_tokens <= old_num_tokens:
            new_bias_present = self.final_logits_bias_present[:, :new_num_tokens]
            new_bias_absent = self.final_logits_bias_absent[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias_present.device)
            new_bias_present = torch.cat([self.final_logits_bias_present, extra_bias], dim=1)
            new_bias_absent = torch.cat([self.final_logits_bias_absent, extra_bias], dim=1)
        self.register_buffer("final_logits_bias_present", new_bias_present)
        self.register_buffer("final_logits_bias_absent", new_bias_absent)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            encoder_outputs=None,
            decoder_present_input_ids=None,
            decoder_absent_input_ids=None,
            decoder_present_attention_mask=None,
            decoder_absent_attention_mask=None,
            decoder_present_cached_states=None,
            decoder_absent_cached_states=None,
            labels_present=None,
            labels_absent=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            **unused,
    ):
        if labels_present is not None or labels_absent is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_present_input_ids=decoder_present_input_ids,
            decoder_absent_input_ids=decoder_absent_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_present_attention_mask=decoder_present_attention_mask,
            decoder_absent_attention_mask=decoder_absent_attention_mask,
            decoder_present_cached_states=decoder_present_cached_states,
            decoder_absent_cached_states=decoder_absent_cached_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        lm_logits_present = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias_present)
        lm_logits_absent = F.linear(outputs[1], self.model.shared.weight, bias=self.final_logits_bias_absent)

        outputs = (lm_logits_present,) + (lm_logits_absent,) + outputs[2:]

        return outputs



















class BartModelTwoDecoders(PretrainedBartModel):
    def __init__(self, config):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder_present = BartDecoder(config, self.shared)
        self.decoder_absent = BartDecoder(config, self.shared)

        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            decoder_present_input_ids=None,
            decoder_absent_input_ids=None,
            encoder_outputs: Optional[Tuple] = None,
            decoder_present_attention_mask=None,
            decoder_absent_attention_mask=None,
            decoder_present_cached_states=None,
            decoder_absent_cached_states=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        if decoder_present_input_ids is None or decoder_absent_input_ids is None:
            #print("None?WTF")
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if not use_cache:
            decoder_present_input_ids, decoder_present_padding_mask, causal_mask_present = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_present_input_ids,
                decoder_padding_mask=decoder_present_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
            decoder_absent_input_ids, decoder_absent_padding_mask, causal_mask_absent = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_absent_input_ids,
                decoder_padding_mask=decoder_absent_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_present_padding_mask, causal_mask_present = None, None
            decoder_absent_padding_mask, causal_mask_absent = None, None

        assert (decoder_present_input_ids is not None and decoder_absent_input_ids is not None)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        assert isinstance(encoder_outputs, tuple)
       # print('present',decoder_present_input_ids)
        #print('absent',decoder_absent_input_ids)

        decoder_present_outputs = self.decoder_present(
            decoder_present_input_ids,
            encoder_outputs[0],
            attention_mask,
            decoder_present_padding_mask,
            decoder_causal_mask=causal_mask_present,
            decoder_cached_states=decoder_present_cached_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )
        decoder_absent_outputs = self.decoder_absent(
            decoder_absent_input_ids,
            encoder_outputs[0],
            attention_mask,
            decoder_absent_padding_mask,
            decoder_causal_mask=causal_mask_absent,
            decoder_cached_states=decoder_absent_cached_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )

        # Attention and hidden_states will be [] or None if they aren't needed
        decoder_present_outputs: Tuple = _filter_out_falsey_values(decoder_present_outputs)
        assert isinstance(decoder_present_outputs[0], torch.Tensor)
        decoder_absent_outputs: Tuple = _filter_out_falsey_values(decoder_absent_outputs)
        assert isinstance(decoder_absent_outputs[0], torch.Tensor)
        encoder_outputs: Tuple = _filter_out_falsey_values(encoder_outputs)
        #print(len(encoder_outputs))
        return decoder_present_outputs + decoder_absent_outputs + encoder_outputs

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.shared)  # make it on the fly


class BartForConditionalGenerationTwoDecoders(PretrainedBartModel):
    base_model_prefix = "model"

    def __init__(self, pre_trained_model):
       # pre_trained_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        super().__init__(pre_trained_model.config)
        base_model = BartModelTwoDecoders(pre_trained_model.config)
        base_model.shared = pre_trained_model.model.shared
        base_model.encoder = pre_trained_model.model.encoder
        base_model.decoder_present = pre_trained_model.model.decoder
        base_model.decoder_absent = copy.deepcopy(pre_trained_model.model.decoder)
        self.model = base_model
        self.register_buffer("final_logits_bias_present", torch.zeros((1, self.model.shared.num_embeddings)))
        self.register_buffer("final_logits_bias_absent", torch.zeros((1, self.model.shared.num_embeddings)))

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        print('resizing............')
        old_num_tokens = self.model.shared.num_embeddings
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self.model.shared = new_embeddings
        self._resize_final_logits_bias(new_num_tokens, old_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int, old_num_tokens: int) -> None:
        if new_num_tokens <= old_num_tokens:
            new_bias_present = self.final_logits_bias_present[:, :new_num_tokens]
            new_bias_absent = self.final_logits_bias_absent[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias_present.device)
            new_bias_present = torch.cat([self.final_logits_bias_present, extra_bias], dim=1)
            new_bias_absent = torch.cat([self.final_logits_bias_absent, extra_bias], dim=1)
        self.register_buffer("final_logits_bias_present", new_bias_present)
        self.register_buffer("final_logits_bias_absent", new_bias_absent)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            encoder_outputs=None,
            decoder_present_input_ids=None,
            decoder_absent_input_ids=None,
            decoder_present_attention_mask=None,
            decoder_absent_attention_mask=None,
            decoder_present_cached_states=None,
            decoder_absent_cached_states=None,
            labels_present=None,
            labels_absent=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            **unused,
    ):
        if labels_present is not None or labels_absent is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_present_input_ids=decoder_present_input_ids,
            decoder_absent_input_ids=decoder_absent_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_present_attention_mask=decoder_present_attention_mask,
            decoder_absent_attention_mask=decoder_absent_attention_mask,
            decoder_present_cached_states=decoder_present_cached_states,
            decoder_absent_cached_states=decoder_absent_cached_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        lm_logits_present = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias_present)
        lm_logits_absent = F.linear(outputs[1], self.model.shared.weight, bias=self.final_logits_bias_absent)

        outputs = (lm_logits_present,) + (lm_logits_absent,) + outputs[2:]
        #print(outputs[-1].shape)

        return outputs
