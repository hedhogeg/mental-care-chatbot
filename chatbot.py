import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
    pad_token='<pad>', mask_token='<mask>')

model = GPT2LMHeadModel.from_pretrained('./model')
dialogue = []

def check_max(tokens):
    '''
    대화를 input으로 넣는 과정에서 max length를 넘는 경우 방지   
    '''
    if len(tokens) < 128:
        return tokens

    else:
        while True:
            usr_idx = tokens.index(2)
            sys_idx = tokens.index(4)
            start = usr_idx if usr_idx > sys_idx else sys_idx
            tokens = tokens[start:]
            if len(tokens) < 128:
                break

        return tokens
    
def get_response(user_input):
    dialogue = []
    with torch.no_grad():
        q = user_input.strip()
        if q[-1] not in ['.', '?', '!']:
            q += '.'
        user = q + "<sys>"
        encoded = tokenizer.encode(user)
        dialogue += encoded
        dialogue = check_max(dialogue)
        input_ids = torch.LongTensor(dialogue).unsqueeze(dim=0)
        output = model.generate(input_ids, max_length=128, num_beams=10, do_sample=False, top_k=50, no_repeat_ngram_size=2, temperature=0.85)
        dialogue = tokenizer.decode(list(output[0])[:-1])
        idx = torch.where(output[0]==tokenizer.encode('<sys>')[0])
        idx = idx[0][-1] + 1
        chatbot = tokenizer.decode(output[0][idx:-1])
    
    return chatbot.strip(), dialogue

