# German GPT-2 model

In this repository we release (yet another) GPT-2 model, that was trained on various texts for German.

The model is meant to be an entry point for fine-tuning on other texts, and it is definitely not as good or "dangerous" as the English GPT-3 model. We do not plan extensive PR or staged releases for this model 😉

**Note**: The model was initially released under an anonymous alias (`anonymous-german-nlp/german-gpt2`) so we now "de-anonymize" it.

More details about GPT-2 can be found in the great [Hugging Face](https://huggingface.co/transformers/model_doc/gpt2.html) documentation.

# Changelog

15.11.2020: Initial release.

# Training corpora

We use pretty much the same corpora as used for training the DBMDZ BERT model, that can be found in [this repository](https://github.com/dbmdz/berts).

Thanks to the awesome Hugging Face team, it is possible to create byte-level BPE with their awesome [Tokenizers](https://github.com/huggingface/tokenizers) library.

With the previously mentioned awesome Tokenizers library we created a 52K byte-level BPE vocab based on the training corpora.

After creating the vocab, we could train the GPT-2 for German on one TPU over the complete training corpus (three epochs).

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
                 tokenizer="dbmdz/german-gpt2", config={'max_length':800})   

text = pipe2("Der Sinn des Lebens ist es")[0]["generated_text"]

print(text)
```

This could output this beautiful text:

```
Der Sinn des Lebens ist es, im Geist zu verweilen, aber nicht in der Welt zu sein, sondern ganz im Geist zu leben.
Die Menschen beginnen, sich nicht nach der Natur und nach der Welt zu richten, sondern nach der Seele,'
```

It is also possible to generate text using the (web-based) inference widget from the [model hub page](https://huggingface.co/dbmdz/german-gpt2).

# Fine-Tuning on other texts

Thanks to the awesome Transformers library, it is also possible to "fine-tune" the model on your own texts. Fine-Tuning can be done with the [language-modeling](https://github.com/huggingface/transformers/tree/master/examples/language-modeling) example from Transformers library

## German GPT-2 fine-tuned on Faust Faust I and II

We fine-tuned our German GPT-2 model on "Faust I and II" from Johann Wolfgang Goethe. These texts can be obtained from [Deutsches Textarchiv (DTA)](http://www.deutschestextarchiv.de/book/show/goethe_faust01_1808). We use the "normalized" version of both texts (to avoid out-of-vocabulary problems with e.g. "ſ")

Fine-Tuning was done for 100 epochs, using a batch size of 4 with half precision on a RTX 3090. Total time was around 12 minutes (it is really fast!).

We also open source this fine-tuned model. Text can be generated with:

```python
from transformers import pipeline

pipe = pipeline('text-generation', model="dbmdz/german-gpt2-faust",
                 tokenizer="dbmdz/german-gpt2-faust", config={'max_length':800})   

text = pipe2("Schon um die Liebe")[0]["generated_text"]

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

## Fine-tuning on German recipes

[Philipp Schmid](https://github.com/philschmid) fine-tuned our model on German recipes - please enjoy the delicious [medium post](https://towardsdatascience.com/fine-tune-a-non-english-gpt-2-model-with-huggingface-9acc2dc7635b) for more details!

# License

All models are licensed under [MIT](LICENSE).

# Huggingface model hub

All models are available on the [Huggingface model hub](https://huggingface.co/dbmdz).

# Contact (Bugs, Feedback, Contribution and more)

For questions about our BERT models just open an issue
[here](https://github.com/stefan-it/german-gpt/issues/new) 🤗

# Acknowledgments

Research supported with Cloud TPUs from Google's TensorFlow Research Cloud (TFRC).
Thanks for providing access to the TFRC ❤️

Thanks to the generous support from the [Hugging Face](https://huggingface.co/) team,
it is possible to download both cased and uncased models from their S3 storage 🤗
