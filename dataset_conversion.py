import re
import spacy

nlp = spacy.load("en_core_web_md")

ENTITY_MAP = {
    "amount": "AMOUNT",
    "currency": "CURRENCY",
    "beneficiary": "BENEFICIARY",
    "account": "FROM_ACCOUNT",
    "method": "PAYMENT_TYPE"
}

def convert_annotated_text(text):
    tokens = []
    labels = []

    pattern = re.compile(r"<START:\s*(\w+)\s*>(.*?)<END>", re.DOTALL)

    idx = 0
    for match in pattern.finditer(text):
        # Text before entity
        pre_text = text[idx:match.start()]
        for t in nlp(pre_text):
            if not t.is_space:
                tokens.append(t.text)
                labels.append("O")

        entity_type = ENTITY_MAP[match.group(1)]
        entity_text = match.group(2).strip()
        entity_tokens = [t.text for t in nlp(entity_text) if not t.is_space]

        for i, tok in enumerate(entity_tokens):
            prefix = "B-" if i == 0 else "I-"
            tokens.append(tok)
            labels.append(prefix + entity_type)

        idx = match.end()

    # Remaining text
    post_text = text[idx:]
    for t in nlp(post_text):
        if not t.is_space:
            tokens.append(t.text)
            labels.append("O")

    return tokens, labels
