from transformers import pipeline, PreTrainedTokenizerFast, BartForConditionalGeneration

def process_text(text):
    tokenizer = PreTrainedTokenizerFast.from_pretrained("ainize/bart-base-cnn")
    model = BartForConditionalGeneration.from_pretrained("ainize/bart-base-cnn")

    sa = pipeline("sentiment-analysis", top_k=None,
                  model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                  device="cpu", padding=True, truncation=True,
                  max_length=10240,
                  verbose=False)

    ea = pipeline("text-classification", model="bdotloh/just-another-emotion-classifier", device="cpu",
                  padding=True, truncation=True, max_length=10240, verbose=False)

    input_ids = tokenizer.encode(text, return_tensors="pt")

    summary_text_ids = model.generate(
        input_ids=input_ids,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        length_penalty=2.0,
        max_length=10240,
        min_length=56,
        num_beams=4,
    )

    sentiment = sa(text)
    emotion = ea(text)
    summary = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)

    return sentiment, emotion, summary
