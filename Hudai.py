'''class BartForConditionalGeneration_separate_decoders(PretrainedBartModel):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig, encoder, decoder):
        super().__init__(config)
        base_model = BartModel(config)
        base_model.encoder = encoder
        base_model.decoder = decoder
        self.model = base_model
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        old_num_tokens = self.model.shared.num_embeddings
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self.model.shared = new_embeddings
        self._resize_final_logits_bias(new_num_tokens, old_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int, old_num_tokens: int) -> None:
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
            self,
            input_ids,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            decoder_cached_states=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            **unused,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
            with labels
            in ``[0, ..., config.vocab_size]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Conditional generation example::

            # Mask filling only works for bart-large
            from transformers import BartTokenizer, BartForConditionalGeneration
            tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
            TXT = "My friends are <mask> but they eat too many carbs."

            model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
            input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
            logits = model(input_ids)[0]

            masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
            probs = logits[0, masked_index].softmax(dim=0)
            values, predictions = probs.topk(5)

            tokenizer.decode(predictions).split()
            # ['good', 'great', 'all', 'really', 'very']
        """
        if "lm_labels" in unused:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                DeprecationWarning,
            )
            labels = unused.pop("lm_labels")

        if labels is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        outputs = (lm_logits,) + outputs[1:]  # Add cache, hidden states and attention if they are here
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # TODO(SS): do we need to ignore pad tokens in labels?
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs

    def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask, use_cache, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        encoder_outputs, decoder_cached_states = past
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "decoder_cached_states": decoder_cached_states,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == 1:
            self._force_token_ids_generation(logits, self.config.bos_token_id)
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(logits, self.config.eos_token_id)
        return logits

    def _force_token_ids_generation(self, scores, token_ids) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0"""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        all_but_token_ids_mask = torch.tensor(
            [x for x in range(self.config.vocab_size) if x not in token_ids],
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        assert len(scores.shape) == 2, "scores should be of rank 2 with shape: [batch_size, vocab_size]"
        scores[:, all_but_token_ids_mask] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        ((enc_out, enc_mask), decoder_cached_states) = past
        reordered_past = []
        for layer_past in decoder_cached_states:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: _reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)

        new_enc_out = enc_out if enc_out is None else enc_out.index_select(0, beam_idx)
        new_enc_mask = enc_mask if enc_mask is None else enc_mask.index_select(0, beam_idx)

        past = ((new_enc_out, new_enc_mask), reordered_past)
        return past

    def get_encoder(self):
        return self.model.encoder

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.model.shared)  # make it on the fly

class BartForConditionalGenerationTwoDecoders_kdsljfklsdjfldsf(nn.Module):
    def __init__(self, pretrainedModel):
        super(BartForConditionalGenerationTwoDecoders, self).__init__()
        self.present_model = BartForConditionalGeneration_separate_decoders(pretrainedModel.config, pretrainedModel.model.encoder, pretrainedModel.model.decoder)
        self.absent_model = BartForConditionalGeneration_separate_decoders(pretrainedModel.config,
                                                                            pretrainedModel.model.encoder,
                                                                            copy.deepcopy(
                                                                                pretrainedModel.model.decoder))

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
        present_outputs = self.present_model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_present_input_ids,
                        decoder_attention_mask=decoder_present_attention_mask, labels=labels_present)'''
