from collections import defaultdict
import json
import numpy as np
import tiktoken

def number_of_assistant_tokens_from_messages(messages):

    encoding = tiktoken.get_encoding('cl100k_base')

    number_of_tokens = 0

    for message in messages:

        if message['role'] == 'assistant':

            number_of_tokens += len(encoding.encode(message['content']))

    return number_of_tokens

def number_of_tokens_from_messages(messages, tokens_per_message = 3, tokens_per_name = 1):

    encoding = tiktoken.get_encoding('cl100k_base')

    number_of_tokens = 0

    for message in messages:

        number_of_tokens += tokens_per_message

        for key, value in message.items():

            number_of_tokens += len(encoding.encode(value))

            if key == 'name':

                number_of_tokens += tokens_per_name

    number_of_tokens += 3

    return number_of_tokens

def print_distribution(values, name):

    print(f'\nDistribution of {name}:')
    print(f'Min/Max: {min(values)}, {max(values)}')
    print(f'Mean/Median: {np.mean(values)}, {np.median(values)}')
    print(f'P5/P95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}') 

def run_check():

    fine_tuning_data_path = './fine_tuning_data/fine_tuning_data.jsonl'

    with open(fine_tuning_data_path, 'r', encoding = 'utf-8') as file:
    
        dataset = [json.loads(line) for line in file]

    print('\nNumber of Examples:', len(dataset))

    errors = defaultdict(int)
    error_lines = defaultdict(list)

    number_of_messages_per_example = []
    convo_lens = []
    assistant_message_lens = []

    max_tokens_per_example = 4096

    target_epochs = 3
    min_default_epochs = 1
    max_defaul_epochs = 25
    min_target_examples = 15
    max_target_examples = 25000

    line_number = 0

    for example in dataset:
    
        line_number += 1
    
        if not isinstance(example, dict):

            errors['Data Type Error'] += 1
            error_lines['Data Type Error'].append(line_number)

            continue

        messages = example.get('messages', None)

        if not messages:

            errors['Missing Messages List'] += 1
            error_lines['Missing Messages List'].append(line_number)
            
            continue

        for message in messages:

            if 'role' not in message or 'content' not in message:
                
                errors['Message Missing Key'] += 1
                error_lines['Message Missing Key'].append(line_number)

            if any(key not in ('role', 'content', 'name', 'function_call') for key in message):
                
                errors['Message Unrecognized Key'] += 1
                error_lines['Message Unrecognized Key'].append(line_number)

            if message.get('role', None) not in ('system', 'user', 'assistant', 'function'):
                
                errors['Unrecognized Role'] += 1
                error_lines['Unrecognized Role'].append(line_number)

            content = message.get('content', None)

            function_call = message.get('function_call', None)

            if (not content and not function_call) or not isinstance(content, str):
                
                errors['Missing Content'] += 1
                error_lines['Missing Content'].append(line_number)

            if not any(message.get('role', None) == 'assistant' for message in messages):

                errors['Missing Assistant Message'] += 1
                error_lines['Missing Assistant Message'].append(line_number)

            number_of_messages_per_example.append(len(messages))
            convo_lens.append(number_of_tokens_from_messages(messages))   
            assistant_message_lens.append(number_of_assistant_tokens_from_messages(messages)) 

    if errors:
    
        print('\nErrors Found!\n')

        for key, value in errors.items():

            print(f'{key}: {value}', '- Lines:', error_lines[key])

    else:
    
        print('\nNo Errors Found!') 

    print_distribution(number_of_messages_per_example, 'Number of Messages per Example')
    print_distribution(convo_lens, 'Number of Total Tokens per Example')
    print_distribution(assistant_message_lens, 'Number of Assistant Tokens per Example')

    number_of_too_long_examples = sum(length > 4096 for length in convo_lens)

    print(f'\n{number_of_too_long_examples} examples may be over the 4096 token limit and will be truncated during fine-tuning')          

    number_of_epochs = target_epochs
    number_of_training_examples = len(dataset)

    if number_of_training_examples * target_epochs < min_target_examples:

        number_of_epochs = min(max_defaul_epochs, min_target_examples // number_of_training_examples)

    elif number_of_training_examples * target_epochs > max_target_examples:

        number_of_epochs = max(min_default_epochs, max_target_examples // number_of_training_examples)

    number_of_billing_tokens_in_dataset = sum(min(max_tokens_per_example, length) for length in convo_lens)

    print(f'The dataset has ~{number_of_billing_tokens_in_dataset} tokens that will be charged for during training')
    print(f'By default, you will train for {number_of_epochs} epochs on this dataset')
    print(f'By default, you will be charged for ~{number_of_epochs * number_of_billing_tokens_in_dataset} tokens\n') 

if __name__ == '__main__':
 
  run_check()        