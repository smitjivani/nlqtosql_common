import os
import sys
from dotenv import load_dotenv

# Load the .env file
# load_dotenv()

# sys.path.insert(1, os.getenv("PROJECT_ROOT"))
# os.environ['HF_HOME'] = os.getenv("HF_CACHE")


import logging
from pathlib import Path
import torch
from .constants import *
from .tecod_utils import (
    convert_sql_string_to_template,
    convert_template_to_ebnf,
    ebnf_to_regex,
    decode_token_ids,
    get_token_offsets,
    get_covering_token_ids,
    get_sub_sqls,
)
import sqlglot
from sqlglot import TokenType
import outlines
import pickle
from tqdm import tqdm
import re
import json


def generate_token_ids_and_save_to_store(*, model, template_id, tokenizer, prompt, sql_query, db_id=None, dataset_name=None, db_path=None, ebnf_type, token_healing=True, token_healing_right=False):
    """
    Generate token IDs for the given SQL query in teacher forcing manner and save them to the specified output path.
    """
    # logging.info(f"Generating token IDs for template: {template_id}")

    text = prompt # prompt with/without chat_template should be handled before passing here

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # prepare template
    # convert sql string to template
    sql_template = convert_sql_string_to_template(sql_string=sql_query,
                                   db_name=db_id,
                                   dataset_name=dataset_name,
                                   db_path=db_path,
                                   mask_literals=False,) # mask_literals=False is used to generate the template for teacher forcing setup, for normal gcd use mask_literals=True

    sql_ebnf, new_rules_for_ebnf = convert_template_to_ebnf(template=sql_template,
                                        remove_aliases=False,
                                        db_id=db_id,
                                        dataset=dataset_name,
                                        db_path=db_path,
                                        type=ebnf_type) # tab_and_col_to_rule_grammar, base_grammar
    

    sql_regex_grammar = ebnf_to_regex(ebnf_str=sql_ebnf, new_rules_for_ebnf=new_rules_for_ebnf)

    # tokenizer for outlines grammar constraint processor
    outlines_tokenizer = outlines.models.TransformerTokenizer(tokenizer)

    grammar_processor = [outlines.processors.RegexLogitsProcessor(
                                sql_regex_grammar,
                                outlines_tokenizer,)]
    
    model_output = model.generate(**model_inputs, 
                                        max_new_tokens=650,
                                        pad_token_id=tokenizer.eos_token_id, 
                                        logits_processor=grammar_processor,
                                        return_dict_in_generate=True,
                                        do_sample=False,
                                        )

    generated_sql, generated_token_ids = decode_token_ids(tokenizer, model_output, model_inputs)

    number_rule = ebnf_to_regex("number_rule", full_sql=False)
    string_rule = ebnf_to_regex("string_rule", full_sql=False)

    # identifying sql partitions around literals
    template_tokens_offsets = get_token_offsets(tokenizer, generated_token_ids) # offsets are [start, end)
    sql_tokens = sqlglot.tokenize(generated_sql, read='sqlite') 
    sql_literal_tokens = [token for token in sql_tokens if token.token_type in [TokenType.NUMBER, TokenType.STRING]]

    sql_literal_indices = []
    literal_extra_texts = []
    # For each literal, find covering token sequence
    for sql_token in sql_literal_tokens:
        literal_span = (sql_token.start, sql_token.end)
        start_idx, end_idx, before_text, after_text = get_covering_token_ids(
            tokenizer,
            generated_token_ids,
            literal_span,
            template_tokens_offsets
        )
        sql_literal_indices.append((start_idx, end_idx))
        literal_extra_texts.append((before_text, after_text))

    input_ids = get_sub_sqls(generated_token_ids, sql_literal_indices)
    sql_literal_types = [token.token_type for token in sql_literal_tokens]

    compiled_template = {
        template_id: {
            'input_ids': input_ids,
            'sql_ebnf': sql_ebnf,
            'sql_literal_types': sql_literal_types,
            'logit_processors': []
        }
    }

    for idx, j in enumerate(sql_literal_types):
        if j == TokenType.NUMBER:
            next_token_id = ""
            prev_token_id = "" # prev token id is used to simulate token healing which will happen naturally with GCD
            
            if token_healing:
                if len(input_ids[idx+1]) > 0:
                    next_token_id = input_ids[idx+1].pop(0)
                if len(input_ids[idx]) > 0:
                    prev_token_id = input_ids[idx].pop()
            elif token_healing_right:
                if len(input_ids[idx+1]) > 0:
                    next_token_id = input_ids[idx+1].pop(0) #rhs token healing

            regex = f"""{re.escape(literal_extra_texts[idx][0])}{number_rule}{re.escape(literal_extra_texts[idx][1])}"""

            regex = f"""{re.escape(tokenizer.decode(prev_token_id)) if prev_token_id != "" else ""}{regex}{re.escape(tokenizer.decode(next_token_id)) if next_token_id != "" else ""}"""
            
        else:
            next_token_id = ""
            prev_token_id = "" # prev token id is used to simulate token healing which will happen naturally with GCD
            
            if token_healing:
                if len(input_ids[idx+1]) > 0:
                    next_token_id = input_ids[idx+1].pop(0)
                if len(input_ids[idx]) > 0:
                    prev_token_id = input_ids[idx].pop()
            elif token_healing_right:
                if len(input_ids[idx+1]) > 0:
                    next_token_id = input_ids[idx+1].pop(0) #rhs token healing

            regex = f"""{re.escape(literal_extra_texts[idx][0])}{string_rule}{re.escape(literal_extra_texts[idx][1])}"""

            regex = f"""{re.escape(tokenizer.decode(prev_token_id)) if prev_token_id != "" else ""}{regex}{re.escape(tokenizer.decode(next_token_id)) if next_token_id != "" else ""}"""

        # regex processor
        processor = outlines.processors.RegexLogitsProcessor(
                        regex,
                        outlines_tokenizer,
                    )
        
        compiled_template[template_id]['logit_processors'].append(processor)

    return compiled_template


def main():
    model_name = XIYAN_SQL_QWENCODER_7B
    device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False, padding_side="left", token=os.getenv("HF_ACCESS_TOKEN"), local_files_only=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=os.getenv("HF_ACCESS_TOKEN"), local_files_only=False, device_map={"": f"{device}"})

    prompts = json.load(open("data/bird/dev/XiYanSQL-QwenCoder-14B-2504_icl-3_gcd.json", 'r'))
    grammar_type = "base_grammar" # Change to 'base_grammar', tab_and_col_to_rule_grammar if needed

    templates = {}

    for prompt in tqdm(prompts[:5]):
        template_id = prompt['idx']
        sql_query = prompt['gold_sql']
        db_id = prompt['db_id']
        dataset_name = 'bird'
        prompt_text = prompt['prompt']
        compiled_template = generate_token_ids_and_save_to_store(
            model=model,
            template_id=template_id,
            tokenizer=tokenizer,
            prompt=prompt_text,
            sql_query=sql_query,
            db_id=db_id,
            ebnf_type=grammar_type,  
            dataset_name=dataset_name,
            token_healing=True,  # Set to True for token healing
            token_healing_right=False  # Set to False for left token healing
        )
        templates.update(compiled_template)
    output_path = Path("release/saved_templates") / f"{dataset_name}" / f"{model_name.split("/")[-1]}" / f"compiled_templates_{grammar_type}.pkl"
    os.makedirs(output_path.parent, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(templates, f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()


