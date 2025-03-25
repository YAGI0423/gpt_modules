
import torch
from torch.nn import functional as F

from torch import Tensor
from torch.nn import Module

import datasets
from datasets import DatasetDict
from datasets import load_dataset

from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from gptModules import models


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Argument Help')
    parser.add_argument('--model', type=str, default='GPT', choices=(
        'GPT',
        'GPT2',
        'ALiBiGPT',
        'LLaMA',
        'DeepSeek',
    ))
    parser.add_argument('--device', type=str, default='CUDA', choices=('CPU', 'CUDA'))
    return parser.parse_args()


#Dataset Function Collecter=====================================
def preprocess_dataset(dataset: datasets) -> datasets:
    def tokenizing(user_text: str, ai_text: str, max_length: int=None) -> tuple[dict, dict]:
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
        def create_ai_token(user_att: list, ai_token: list):
            user_len = len(user_att)
            ai_len = max_length - user_len - 1 #max_len - user_len - [EOS](1)

            return [-100] * user_len + ai_token[:ai_len] + EOS_TOKEN
        

        EOS_TOKEN = tokenizer.encode('[EOS]') #[50258]

        user_text = ['user: ' + text for text in user_text]
        ai_text = ['ai: ' + text for text in ai_text]

        user_tokens = tokenizer(user_text, max_length=max_length-1, truncation=True)
        ai_tokens = tokenizer(ai_text)['input_ids']

        #Create ai_token
        ai_tokens = [create_ai_token(*x) for x in zip(user_tokens['attention_mask'], ai_tokens)]

        return {
            'input_idx': user_tokens['input_ids'],
            'attention_mask': user_tokens['attention_mask'],
            'label': ai_tokens,
        }
    
    ds = dataset.map(lambda x: tokenizing(x['instruction'], x['response'], max_length=MAX_SEQ_LEN), batched=True)
    ds = ds.rename_column('input_idx', 'input_ids')
    ds = ds.remove_columns(('instruction', 'context', 'response', 'category'))
    return ds

    
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


def split_datasets(dataset: datasets) -> datasets:
    train_test = dataset['train'].train_test_split(test_size=0.2)
    test_valid = train_test['test'].train_test_split(test_size=0.5)

    return DatasetDict({
        'train': train_test['train'],
        'test': test_valid['train'],
        'valid': test_valid['test'],
    })
#End============================================================


def predict(model: Module, x: dict, device: str='cuda:0') -> Tensor:
    input_ids, att_mask, y = x['input_ids'], x['attention_mask'], x['label']
    input_ids, att_mask, y = input_ids.to(device), att_mask.to(device), y.to(device)
    y = y.view(-1)

    predict = model(input_ids, att_mask)

    if type(model) is models.DeepSeek:
        y_hat, aux_loss = predict
        y_hat = y_hat.view(-1, VOCAB_SIZE)
        return F.cross_entropy(y_hat, y, ignore_index=-100) + aux_loss * 0.0001
    else:
        y_hat = predict.view(-1, VOCAB_SIZE)
        return F.cross_entropy(y_hat, y, ignore_index=-100)


def create_graph(title: str, train_losses: list, val_losses: list, save_path: str) -> None:
        def set_general_option() -> None:
            plt.ylabel('Cross Entropy Loss')
            plt.legend()
            plt.grid()


        plt.figure(figsize=(7, 7))
        plt.suptitle(title, fontsize=13, fontweight='bold')
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.92, wspace=0.175, hspace=0.25)
                
        #Train
        plt.subplot(2, 1, 1)
        plt.plot(train_losses, color='blue', label='train')
        set_general_option()

        #Validation
        plt.subplot(2, 1, 2)
        plt.plot(range(1, len(val_losses)+1), val_losses, color='red', label='validation')
        plt.xlim(0, len(val_losses))
        plt.xlabel('Epoch')
        set_general_option()

        
        #Save Graph
        plt.savefig(save_path)


if __name__ == '__main__': 
    args = get_args()
    MODEL_PATH = './saved_models/'
    GRAPH_PATH = './figures/'

    EPOCH = 8
    BATCH_SIZE = 8
    MAX_SEQ_LEN = 1000
    
    device = args.device.lower()

    
    #Set Tokenizer
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
    ds = preprocess_dataset(ds)
    ds = split_datasets(ds) #train, val, test

    STEP_PER_EPOCH = ds['train'].num_rows // BATCH_SIZE


    #Set Model
    model_args = {
        'vocab_size': VOCAB_SIZE,
        'n_layers': 9,
        'n_heads': 10,
        'd_model': 560,
        'd_ff': 2304,
        'max_seq_length': MAX_SEQ_LEN,
    }

    if args.model == 'GPT':
        model = models.GPT(**model_args)
    elif args.model == 'GPT2':
        model = models.GPT2(**model_args)
    elif args.model == 'ALiBiGPT':
        model = models.ALiBiGPT(**model_args)
    elif args.model == 'LLaMA':
        model_args['n_groups'] = 5 #GQA Group Heads
        model_args['base'] = 500_000 #RoPE(theta)

        model = models.LLaMA(**model_args)
    elif args.model == 'DeepSeek':
        model_args['n_shared'] = 1
        model_args['n_expert'] = 4
        model_args['d_ff'] = 576 #d_ff // n_expert
        model_args['top_k'] = 2
        model_args['d_kv_comp'] = 12
        model_args['d_rope'] = 14
        model_args['rope_base'] = 10_000
        model = models.DeepSeek(**model_args)
    else:
        raise Exception('Invalid `model` name')
    
    model.to(device)


    #Set Train
    optim = torch.optim.AdamW(model.parameters(), lr=0.000022, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=0.00022, steps_per_epoch=STEP_PER_EPOCH, epochs=EPOCH)

    
    train_losses = list()
    val_losses = list()
    

    for e in range(EPOCH):
        train_loader = DataLoader(
            ds['train'],
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=custom_collate_fn
        )
        train_loader = tqdm(train_loader)


        test_loader = DataLoader(
            ds['test'],
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=custom_collate_fn
        )
        

        val_loader = DataLoader(
            ds['valid'],
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=custom_collate_fn
        )
        

        model.train()
        for train_data in train_loader:
            optim.zero_grad()
            train_loss = predict(model, train_data, device=device)

            train_loss.backward()
            optim.step()
            scheduler.step()

            train_loss = train_loss.item()

            desc = f'train loss: {train_loss:.3f}'
            train_loader.set_description(desc)
            train_losses.append(train_loss)
            
        
        
        #valid 
        model.eval()
        val_buff = list()
        with torch.no_grad():
            for val_data in val_loader:
                val_loss = predict(model, val_data, device=device).item()
                val_buff.append(val_loss)
        
        val_loss = sum(val_buff) / len(val_buff)
        val_losses.append(val_loss)


        
   
    #Test
    model.eval()
    with torch.no_grad():
        for test_data in test_loader:
            test_loss = predict(model, test_data, device=device).item()
    
    #Print Test Loss
    print('< TEST LOSS >'.center(100, '='))
    print(f'Test Loss: {test_loss:.3f}')
    print('=' * 100)
    
    '''
    GPT         Test Loss: 3.106
    GPT2        Test Loss:
    ALiBi GPT   Test Loss:
    LLaMA       Test Loss:
    DeepSeek    Test Loss:
    '''


    #Save Model
    torch.save(model.state_dict(), MODEL_PATH + f'{args.model}_{EPOCH}EPOCH.pt') #path: './saved_models/DeepSeek_2.pt'


    #Save Graph
    create_graph(
        title=f'{args.model} Train Result',
        train_losses=train_losses,
        val_losses=val_losses,
        save_path=f'{GRAPH_PATH}{args.model}_{EPOCH}.png', #path: './figures/DeepSeek_2.png
    )
    

    



