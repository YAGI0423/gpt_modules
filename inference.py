import torch

import argparse

from transformers import GPT2Tokenizer
from gptModules import models



def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Argument Help')
    parser.add_argument('--prompt', type=str, default='What is investment banking?')
    parser.add_argument('--model', type=str, default='GPT', choices=(
        'GPT',
        'GPT2',
        'ALiBiGPT',
        'LLaMA',
        'DeepSeek',
    ))
    parser.add_argument('--device', type=str, default='CUDA', choices=('CPU', 'CUDA'))
    return parser.parse_args()



if __name__ == '__main__': 
    args = get_args()
    MODEL_PATH = './saved_models/'
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
    
    
    model.load_state_dict(torch.load(f'{MODEL_PATH}{args.model}_8EPOCH.pt', weights_only=True))
    model.to(device)

    model.eval()

    prompt = 'user: ' + args.prompt + ' ai: '
    x = tokenizer.encode(prompt, return_tensors='pt').to(device)

    eos_ids = tokenizer.encode('[EOS]')[0]


    for _ in range(100):
        if type(model) is models.DeepSeek:
            predict, _ = model(x)
        else:
            predict = model(x)
        predict_token = predict[0, -1, :].max(dim=-1, keepdim=True).indices
        
        if predict_token.item() == eos_ids:
            break

        x = torch.concat((x, predict_token.view(1, 1)), dim=-1)



    print('\n')
    print(f'< {args.model} Inference >'.center(100, '='))
    print('user:', args.prompt)
    print(tokenizer.decode(x[0]))
    print('=' * 100)

