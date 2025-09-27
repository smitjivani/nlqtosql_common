import os
import sys
from dotenv import load_dotenv

# Load the .env file
# load_dotenv()
# sys.path.insert(1, os.getenv("PROJECT_ROOT"))
# os.environ['HF_HOME'] = os.getenv("HF_CACHE")
# os.environ['CURL_CA_BUNDLE'] = ''

import pickle
import logging
from .constants import *
# from utils.common_utils import *
# from time import process_time
from tqdm.auto import tqdm
from pathlib import Path
import torch
import json
from .tecod_utils import remove_trailing_kv_cache, remove_trailing_eos_tensor, truncate_kv_cache


def partitioned_decoding(model, tokenizer, prompt, template_id, template, device):
    """
    Perform partitioned decoding for the given prompts and predictions using the specified model and tokenizer.
    """
    sql_literal_types = template['sql_literal_types']
    logit_processors = template['logit_processors']
    assert len(sql_literal_types) == len(logit_processors), "Mismatch between sql_literal_types and logit_processors length"
    input_ids = template['input_ids']
    model_kwargs = {}

    past_key_values = None
    llm_input = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    prompt_token_count = llm_input.shape[1]

    for idx, lit in enumerate(sql_literal_types):
        processor = logit_processors[idx].copy()

        if past_key_values is not None:
            model_kwargs["past_key_values"] = past_key_values

        if len(input_ids[idx]) > 0:
            llm_input = torch.cat((llm_input, torch.tensor(input_ids[idx]).view(1, -1).to(device)), dim=1)
        else:
            # handles the case where the input_ids is empty so we need to ensure kv cache has 1 seq less than the input_ids otherwise transformers library will throw an error
            past_key_values = remove_trailing_kv_cache(past_key_values, model.__class__.__name__)
            model_kwargs["past_key_values"] = past_key_values

        # crrate attention mask
        attention_mask = torch.ones(llm_input.shape, dtype=torch.long).to(device)


        outputs = model.generate(**{"input_ids": llm_input, "attention_mask": attention_mask},
                                pad_token_id=tokenizer.eos_token_id, 
                                max_new_tokens=100, 
                                logits_processor=[processor],
                                return_dict_in_generate=True,
                                do_sample=False,
                                **model_kwargs
                                )


        # Remove only trailing `eos_token_id`
        output_token_ids = remove_trailing_eos_tensor(outputs["sequences"][0, llm_input.shape[1]:], tokenizer.eos_token_id)

        # kv caching
        past_key_values = truncate_kv_cache(
            outputs["past_key_values"], 
            llm_input.shape[1] + len(output_token_ids),
            model.__class__.__name__
        )

        # add output token ids to the llm_input
        llm_input = torch.cat((llm_input, output_token_ids.view(1, -1).to(device)), dim=1)

    if  len(input_ids[-1]) > 0:
        llm_input = torch.cat((llm_input, torch.tensor(input_ids[-1]).view(1, -1).to(device)), dim=1)
            
    output_token_ids = llm_input[0][prompt_token_count:]
    sql_text = tokenizer.decode(output_token_ids, skip_special_tokens=True)

    return sql_text

def main():
    model_name = XIYAN_SQL_QWENCODER_7B
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False, padding_side="left", token=os.getenv("HF_ACCESS_TOKEN"), local_files_only=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=os.getenv("HF_ACCESS_TOKEN"), local_files_only=False, device_map={"": f"{device}"})
    dataset_name = 'bird'
    grammar_type = "tab_and_col_to_rule_grammar" # Change to 'base_grammar', tab_and_col_to_rule_grammar if needed

    prompts = json.load(open("data/bird/dev/XiYanSQL-QwenCoder-14B-2504_icl-3_gcd.json", 'r'))
    template_path = Path("release/saved_templates") / f"{dataset_name}" / f"{model_name.split("/")[-1]}" / f"compiled_templates_{grammar_type}.pkl"
    with open(template_path, 'rb') as f:
        templates = pickle.load(f)

    generated_sqls = []

    for prompt in tqdm(prompts[:5]):
        template_id = prompt['idx']
        template = templates[template_id]
        prompt_text = prompt['prompt']
        generated_sql = partitioned_decoding(
            model=model,
            template_id=template_id,
            tokenizer=tokenizer,
            prompt=prompt_text,
            device=device,
            template=template
        )
        generated_sqls.append({
            'idx': template_id,
            'generated_sql': generated_sql,
            'prompt': prompt_text,
            'db_id': prompt['db_id'],
            'gold_sql': prompt['gold_sql']
        })

    output_path = Path("release/saved_templates") / f"{dataset_name}" / f"{model_name.split("/")[-1]}" / f"partitioned_decoding_output_sqls_{grammar_type}.json"
    os.makedirs(output_path.parent, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(generated_sqls, f, indent=4)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
