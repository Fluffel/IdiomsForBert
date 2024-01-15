from transformers import BertModel, BertTokenizer

if __name__ == '__main__':
    print("Loading model...")
    model = BertModel.from_pretrained("bert-base-uncased")
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text = "Peter died"
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    print(output)