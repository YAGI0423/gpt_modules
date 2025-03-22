
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
    BATCH_SIZE = 4

    '''
    >>> DatasetDict({
        train: Dataset({
            features: ['instruction', 'context', 'response', 'category'],
            num_rows: 15011
        })
    })
    '''
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({
        'pad_token': '[PAD]',
        'cls_token': '[EOS]',
        'mask_token': '[MASK]',
    })


    ds = load_dataset(
        'databricks/databricks-dolly-15k',
        cache_dir='./dataset/',
    )

    user = 'User: ' + ds['train']['instruction'][0]
    ai = 'AI: ' + ds['train']['response'][0]
    

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

        user_tokens = tokenizer(user_text)
        ai_tokens = tokenizer(ai_text)['input_ids']
        

        #Create ai_token
        create_ai_token = lambda user_att, ai_token: [-100] * len(user_att) + ai_token
        ai_tokens = [create_ai_token(*x) for x in zip(user_tokens['attention_mask'], ai_tokens)]

        return {
            'input_idx': user_tokens['input_ids'],
            'attention_mask': user_tokens['attention_mask'],
            'label': ai_tokens,
        }

    
    def renameing(dataset, name: str) -> any:
        ds = dataset.rename_column('input_ids', f'{name}_input_ids')
        ds = ds.rename_column('attention_mask', f'{name}_attention_mask')
        return ds



    
    #Instructnion
    ds = ds.map(lambda x: tokenizing(x['instruction'], x['response']), batched=True)

    print(ds)
    raise

    ds = renameing(ds, 'ins')

    #Response
    ds = ds.map(lambda x: tokenizing(x['instruction'], x['response']), batched=True)
    ds = renameing(ds, 'res')

    ds = ds.remove_columns(('instruction', 'context', 'response', 'category'))
    print(ds)

    # ds = ds.map(lambda x: tokenizing(x['instruction']), batched=True)
    
    # print(ds)
    raise
    train_test_set = ds['train'].train_test_split(test_size=0.2)
    val_set = train_test_set['test'].train_test_split(test_size=0.5)

    ds = DatasetDict({
        'train': train_test_set['train'],
        'valid': val_set['train'],
        'test': val_set['test'],
    })

    print(ds, len(ds))

    raise
    text = dataset['test']['text'][0]

    

    
    print(text)
    print(len(tokenizer))
    print(tokenizer(text))
    raise

    
    #Create Vocabulary
    TEXT.build_vocab(trainset, min_freq=5) #5회 이상 등장 단어만 추가
    LABEL.build_vocab(trainset)

    vocab_size = len(TEXT.vocab)
    n_class = 2
    

    #Split Validation
    trainset, valset = trainset.split(0.8)
    

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (trainset, valset, testset), 
        batch_size=BATCH_SIZE,
        shuffle=True,
        repeat=False,
    )
    
    for x, y in test_iter:
        print(x, y)
        raise
    raise

    model = GPT(
        vocab_size=15,
        n_layers=3,
        n_heads=2,
        d_model=16,
        d_ff=64,
        max_seq_length=25,
    )



