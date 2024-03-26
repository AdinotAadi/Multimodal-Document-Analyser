from transformers import pipeline, PreTrainedTokenizerFast, BartForConditionalGeneration

def process_text(text):
    text = text.replace("\n", " ")
    sentiments = []
    emotions = []
    summaries = []
    tokenizer = PreTrainedTokenizerFast.from_pretrained("ainize/bart-base-cnn")
    model = BartForConditionalGeneration.from_pretrained("ainize/bart-base-cnn")

    sa = pipeline("sentiment-analysis", top_k=None,
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device="cpu", padding=True, truncation=True,
                max_length=1024,
                verbose=False)

    ea = pipeline("text-classification", model="bdotloh/just-another-emotion-classifier", device="cpu", padding=True, truncation=True, max_length=1024, verbose=False)

    for i in range(0, len(text), 500):
        batch_text = text[i:i+500]

        input_ids = tokenizer.encode(batch_text, return_tensors="pt")

        summary_text_ids = model.generate(
            input_ids=input_ids,
            bos_token_id=model.config.bos_token_id,
            eos_token_id=model.config.eos_token_id,
            length_penalty=2.0,
            max_length=1024,
            min_length=56,
            num_beams=4,
        )
        sentiments.append(sa(batch_text))
        emotions.append(ea(batch_text))
        summaries.append(tokenizer.decode(summary_text_ids[0], skip_special_tokens=True))

    combined_sentiments = [sent for batch_sentiments in sentiments for sent in batch_sentiments]
    combined_emotions = [emo for batch_emotions in emotions for emo in batch_emotions]
    combined_summary = "\n".join(summaries)

    return combined_sentiments[0][0], combined_emotions[0], combined_summary