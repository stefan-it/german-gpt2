# German GPT-2 model

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4275046.svg)](https://doi.org/10.5281/zenodo.4275046)

In this repository we release (yet another) GPT-2 model, that was trained on various texts for German.

The model is meant to be an entry point for fine-tuning on other texts, and it is definitely not as good or "dangerous" as the English GPT-3 model. We do not plan extensive PR or staged releases for this model 😉

**Note**: The model was initially [released](https://huggingface.co/anonymous-german-nlp/german-gpt2) under an anonymous alias (`anonymous-german-nlp/german-gpt2`) so we now "de-anonymize" it.

More details about GPT-2 can be found in the great [Hugging Face](https://huggingface.co/transformers/model_doc/gpt2.html) documentation.

# Changelog

17.08.2021: Public release of re-trained version of our German GPT-2 model with better results.

15.11.2020: Initial release. Please use the tag v1.0 for [this older version](https://huggingface.co/dbmdz/german-gpt2/tree/v1.0).

# Training corpora

We use pretty much the same corpora as used for training the DBMDZ BERT model, that can be found in [this repository](https://github.com/dbmdz/berts).

Thanks to the awesome Hugging Face team, it is possible to create byte-level BPE with their awesome [Tokenizers](https://github.com/huggingface/tokenizers) library.

With the previously mentioned awesome Tokenizers library we created a 50K byte-level BPE vocab based on the training corpora.

After creating the vocab, we could train the GPT-2 for German on a v3-8 TPU over the complete training corpus (20 epochs).

# Training details

We use the JAX/FLAX integration from Transformers to re-train a better version of our GPT-2 model. The following hyperparameters were used:

```bash
./run_clm_flax.py \
    --output_dir="./l" \
    --model_type="gpt2" \
    --config_name="./" \
    --tokenizer_name="./" \
    --do_train --do_eval \
    --block_size="512" \
    --per_device_train_batch_size="64" \
    --per_device_eval_batch_size="64" \
    --learning_rate="5e-3" --warmup_steps="1000" \
    --adam_beta1="0.9" --adam_beta2="0.98" --weight_decay="0.01" \
    --overwrite_output_dir \
    --num_train_epochs="20" \
    --logging_steps="500" \
    --save_steps="2500" \
    --eval_steps="2500" \
```

More details can be found in the [Transformers documentation](https://github.com/huggingface/transformers/blob/master/examples/flax/language-modeling/README.md#train-model-1).

# Using the model

The model itself can be used in this way:

```python
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")

model = AutoModelWithLMHead.from_pretrained("dbmdz/german-gpt2")
```

However, text generation is a bit more interesting, so here's an example that shows how to use the great Transformers *Pipelines* for generating text:

```python
from transformers import pipeline

pipe = pipeline('text-generation', model="dbmdz/german-gpt2",
                 tokenizer="dbmdz/german-gpt2")

text = pipe("Der Sinn des Lebens ist es", max_length=100)[0]["generated_text"]

print(text)
```

This could output this beautiful text:

```
Der Sinn des Lebens ist es, im Geist zu verweilen, aber nicht in der Welt zu sein, sondern ganz im Geist zu leben.
Die Menschen beginnen, sich nicht nach der Natur und nach der Welt zu richten, sondern nach der Seele,'
```

It is also possible to generate text using the (web-based) inference widget from the [model hub page](https://huggingface.co/dbmdz/german-gpt2).

# Fine-Tuning on other texts

Thanks to the awesome Transformers library, it is also possible to "fine-tune" the model on your own texts.
Fine-Tuning can be done with the [language-modeling](https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling)
example from Transformers library.

Feel free to open a PR to include your fine-tuned model here!

## German GPT-2 fine-tuned on Faust I and II

We fine-tuned our German GPT-2 model (v1.0 version) on "Faust I and II" from Johann Wolfgang Goethe. These texts can be obtained from [Deutsches Textarchiv (DTA)](http://www.deutschestextarchiv.de/book/show/goethe_faust01_1808). We use the "normalized" version of both texts (to avoid out-of-vocabulary problems with e.g. "ſ")

Fine-Tuning was done for 100 epochs, using a batch size of 4 with half precision on a RTX 3090. Total time was around 12 minutes (it is really fast!).

We also open source this fine-tuned model. Text can be generated with:

```python
from transformers import pipeline

pipe = pipeline('text-generation', model="dbmdz/german-gpt2-faust",
                 tokenizer="dbmdz/german-gpt2-faust")

text = pipe("Schon um die Liebe", max_length=100)[0]["generated_text"]

print(text)
```

and could output:

```
Schon um die Liebe bitte ich, Herr! Wer mag sich die dreifach Ermächtigen?
Sei mir ein Held!
Und daß die Stunde kommt spreche ich nicht aus.
Faust (schaudernd).
Den schönen Boten finde' ich verwirrend;
```

It is also possible to generate text using the (web-based) inference widget from the [model hub page](https://huggingface.co/dbmdz/german-gpt2-faust).

## Fine-tuning on German recipes

[Philipp Schmid](https://github.com/philschmid) fine-tuned our model (v1.0 version) on German recipes - please enjoy the delicious
[medium post](https://towardsdatascience.com/fine-tune-a-non-english-gpt-2-model-with-huggingface-9acc2dc7635b) for more details!

## Fine-tuning on German medical reviews
A detailed [blog post](https://data-dive.com/finetune-german-gpt2-on-tpu-transformers-tensorflow-for-text-generation-of-reviews) fine-tunes the Tensorflow
version of our model on a large data set of German medical reviews. After training, the model can be prompted to generate positive or negative reviews.

# License

All models are licensed under [MIT](LICENSE).

# Huggingface model hub

All models are available on the [Huggingface model hub](https://huggingface.co/dbmdz).

# Contact (Bugs, Feedback, Contribution and more)

For questions about our GPT-2 models just open an issue
[here](https://github.com/stefan-it/german-gpt/issues/new) 🤗

# Citation

You can use the following BibTeX entry for citation:

```bibtex
@software{stefan_schweter_2020_4275046,
  author       = {Stefan Schweter},
  title        = {German GPT-2 model},
  month        = nov,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.4275046},
  url          = {https://doi.org/10.5281/zenodo.4275046}
}
```

# Acknowledgments

Research supported with Cloud TPUs from Google's TensorFlow Research Cloud (TFRC).
Thanks for providing access to the TFRC ❤️

Thanks to the generous support from the [Hugging Face](https://huggingface.co/) team,
it is possible to download both cased and uncased models from their S3 storage 🤗
