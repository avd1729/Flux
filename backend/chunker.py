import nltk
nltk.download('punkt')

def split_text_into_chunks(text, chunk_size=500, overlap=100):
    
    sentences = nltk.sent_tokenize(text)
    out = []
    curr = ""
    for sentence in sentences:
        if(len(curr) + len(sentence) <= chunk_size):
            curr += " " + sentence
        else:
            out.append(curr.strip())
            # start a new chunk
            curr = " ".join(
                # we can take last few sentences for overlap
                sentences[
                    max(0, sentences.index(sentence) - 2) : sentences.index(sentence) + 1
                ]
            )
    
    if curr.strip():
        out.append(curr.strip())

    return out