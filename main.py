
import torch
from torch import nn

from gptModules import layers

from torch import Tensor
from torch.nn import Module
from torch.nn import Linear
from torch.nn import Dropout



from datasets import DatasetDict
from datasets import load_dataset
from transformers import GPT2Tokenizer

class GPT(Module):
    def __init__(self, vocab_size: int, n_layers: int, n_heads: int, 
                 d_model: int, d_ff: int, max_seq_length: int, dropout: float=0.1):
        super(GPT, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.embedding = layers.Embeddings(max_seq_length, d_model)

        self.dropout = Dropout(dropout)

        #Transformer Blocks(Decoder Blocks)
        self.layers = nn.ModuleList([
            layers.TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        self.out_linear = Linear(d_model, vocab_size)



    def forward(self, x: Tensor, attention_mask: Tensor=None) -> Tensor:

        out = self.token_embedding(x)
        out = self.embedding(out)
        out = self.dropout(out)

        
        #Transformer Decoder Block
        for layer in self.layers:
            out = layer(out, attention_mask)
        
        out = self.out_linear(out)
        return out


if __name__ == '__main__': 
    def tokenizing(user_text: str, ai_text: str) -> tuple[dict, dict]:
        '''
        >>> user_token = {
            'input_ids': [int, ...],
            'attention_mask': [int, ...],
        }
        ai_token = {
            'input_ids': [int, ...],
            'attention_mask': [int, ...],
        }

        '''
        EOS_TOKEN = tokenizer.encode('[EOS]') #[50258]

        user_tokens = tokenizer(user_text)
        ai_tokens = tokenizer(ai_text)['input_ids']

        #Create ai_token
        create_ai_token = lambda user_att, ai_token: [-100] * len(user_att) + ai_token + EOS_TOKEN
        ai_tokens = [create_ai_token(*x) for x in zip(user_tokens['attention_mask'], ai_tokens)]

        return {
            'input_idx': user_tokens['input_ids'],
            'attention_mask': user_tokens['attention_mask'],
            'label': ai_tokens,
        }

    
    def custom_collate_fn(sample: list[dict]) -> dict[Tensor, Tensor, Tensor]:
        '''
        >>> sample = [
            {
                'input_ids': list,
                'attention_mask': list,
                'label': list,
            },
            ...
        ]
        return {
            'input_ids': list,
            'attention_mask': list,
            'label': list,
        }
        '''
        #List To Dict
        sample_dict = {
            'input_ids': list(),
            'attention_mask': list(),
            'label': list(),
        }

        max_len = 0

        for x in sample:
            sample_dict['input_ids'].append(x['input_ids'])
            sample_dict['attention_mask'].append(x['attention_mask'])
            sample_dict['label'].append(x['label'])

            if (y_len := len(x['label'])) > max_len:
                max_len = y_len


        for k, values in sample_dict.items():
            for i, v in enumerate(values):
                sample_dict[k][i] = sample_dict[k][i] + [0] * (max_len - len(v))
            sample_dict[k] = torch.tensor(sample_dict[k])

        sample_dict['label'] = sample_dict['label']#[:, :, None] #Batch x seq x 1
        return sample_dict

    


    BATCH_SIZE = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({
        'pad_token': '[PAD]',
        'eos_token': '[EOS]',
        'mask_token': '[MASK]',
    })

    VOCAB_SIZE = tokenizer.vocab_size + 3 #[PAD], [EOS], [MASK]


    ds = load_dataset(
        'databricks/databricks-dolly-15k',
        cache_dir='./dataset/',
    )
    
    #Instructnion
    ds = ds.map(lambda x: tokenizing(x['instruction'], x['response']), batched=True)
    ds = ds.rename_column('input_idx', 'input_ids')
    ds = ds.remove_columns(('instruction', 'context', 'response', 'category'))


    #get max_seq_length
    MAX_SEQ_LEN = max(len(tk) for tk in ds['train']['label'])


    dataLoader = torch.utils.data.DataLoader(ds['train'], batch_size=BATCH_SIZE, collate_fn=custom_collate_fn) #shuffle=True, 
    

    model = GPT(
        vocab_size=VOCAB_SIZE,
        n_layers=3,
        n_heads=2,
        d_model=16,
        d_ff=64,
        max_seq_length=MAX_SEQ_LEN,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for data in dataLoader:
        input_ids, att_mask, y = data['input_ids'], data['attention_mask'], data['label']
        
        input_ids, att_mask, y = input_ids.to(device), att_mask.to(device), y.to(device)

        out = model(input_ids, att_mask)

        out = out.view(-1, VOCAB_SIZE) #([batch * seq] x vocab_size)
        y = y.view(-1) #([batch x seq], )
        
        loss = loss_fn(out, y)
        print(loss)
        raise
        
        


        raise

    raise


    



