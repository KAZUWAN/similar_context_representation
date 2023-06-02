import torch

# gain word_emb
def Bert_get_hidden(input_id,attention_mask,model):

    tokens_tensor=torch.tensor([input_id]) 
    attention_tensor=torch.tensor([attention_mask])

    hidden_states=model(tokens_tensor,attention_tensor,output_hidden_states=True)
    
    return hidden_states