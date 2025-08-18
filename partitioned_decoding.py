import os
import sys
from dotenv import load_dotenv

# Load the .env file
load_dotenv()
sys.path.insert(1, os.getenv("PROJECT_ROOT"))
os.environ['HF_HOME'] = os.getenv("HF_CACHE")
# os.environ['CURL_CA_BUNDLE'] = ''

import pickle
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from constants import *
from utils.common_utils import *
# from time import process_time
from tqdm import tqdm
from pathlib import Path
import torch
import json
from release.tecod_utils import *
import contextlib
from tqdm import tqdm


method_output_path_map = {
    "tf_flexible": "tf_flexible",
    "tf_tight": "tf_tight",
    "flexible_tokenizer_guided": "tg_tf_flexible",
    "tight_tokenizer_guided": "tg_tf_tight"
}

@contextlib.contextmanager
def time_block(name, timer_dict=None):
    """Context manager for timing code blocks with CUDA events"""
    if timer_dict is None:
        timer_dict = {}
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start_event.record()
    
    try:
        yield
    finally:
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        
        if name in timer_dict:
            timer_dict[name] += elapsed_time
        else:
            timer_dict[name] = elapsed_time

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
        processor = logit_processors[idx]

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
                                # output_scores=True,
                                # output_logits=True,
                                # output_hidden_states=True,
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

# def main(model_key, device, method, token_healing, token_healing_right, dataset, variant):
    
#     TOKENIZER_GUIDED = False
#     method_output_path = method_output_path_map[method]

#     if "tokenizer_guided" in method:
#         method = "tf_" + method.split("_")[0]
#         TOKENIZER_GUIDED = True
        

#     preds = pickle.load(open(Path(f'{TEMPLATE_PREDS}/{dataset}/{model_key}/{method}/{variant}/out_token_ids.pkl'), 'rb'))

#     # Load model and tokenizer
#     tokenizer = load_tokenizer(model_id_mapping[model_key])
#     model = load_model(model_id_mapping[model_key], device)
#     model.e1 = 0
#     model.e2 = 0
#     model.e3 = 0
#     timers = {}
#     token_stats = []
#     token_stats_columns = ["#non_literal_token", "#number_token", "#string_token", "#literal_token", "#number_literal", "#string_literal", "#total_literals"]

#     # load prompt json
#     prompts = load_prompt(model.__class__.__name__, dataset)

#     outlines_tokenizer = outlines.models.TransformerTokenizer(
#         AutoTokenizer.from_pretrained(model_id_mapping[model_key], local_files_only=True)
#     )

#     sql_number_rule = ebnf_to_regex("sql_number_rule", full_sql=False, variant=variant)
#     string_rule = ebnf_to_regex("string_rule", full_sql=False, variant=variant)

#     logit_processors = []

#     responses = {}
#     errors = []
#     out_ids = {}
#     token_counter = 0
#     literal_counter = 0
#     model_kwargs = {}

#     start_event3 = torch.cuda.Event(enable_timing=True)
#     end_event3 = torch.cuda.Event(enable_timing=True)

#     torch.cuda.synchronize()

#     start_event3.record()

#     for i, (prompt, pred) in tqdm(enumerate(zip(prompts, preds)), total=len(prompts)):
#         try:
#             if i > 0:
#                 exit(0)  # for testing purposes, remove this line in production
#             query_token_stats = [0, 0, 0, 0, 0, 0, 0]
#             current_processors = {}
#             with time_block(f"prompt_{i}_tokenization", timers):
#                 prompt = prompt['prompt']
#                 llm_input = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
#                 prompt_token_count = llm_input.shape[1]
#                 template_tokens = pred["output_token_ids"]
#                 past_key_values = None
#                 # decoded string
#                 template_sql_string = tokenizer.decode(template_tokens, skip_special_tokens=True)

#             with time_block(f"template_{i}_validation", timers):
#                 # check if the template has -1 meaning it is not a valid template
#                 if -1 in template_tokens:
#                     responses.update({str(i): ""})
#                     errors.append(f"{i}: Invalid template")
#                     continue

#             with time_block(f"token_{i}_offset_calculation", timers):
#                 template_tokens_offsets = get_token_offsets(tokenizer, template_tokens) # offsets are [start, end)

#             with time_block(f"input_{i}_sql_parsing", timers):
#                 # get sql literal tokens
#                 sql_tokens = sqlglot.tokenize(template_sql_string, read='sqlite') 
#                 sql_literal_tokens = [token for token in sql_tokens if token.token_type in [TokenType.NUMBER, TokenType.STRING]]

#             with time_block(f"literal_{i}_partition_calculation", timers):
#                 sql_literal_indices = []
#                 literal_extra_texts = []
#                 # For each literal, find covering token sequence
#                 for sql_token in sql_literal_tokens:
#                     literal_span = (sql_token.start, sql_token.end)
#                     start_idx, end_idx, before_text, after_text = get_covering_token_ids(
#                         tokenizer,
#                         template_tokens,
#                         literal_span,
#                         template_tokens_offsets
#                     )
#                     sql_literal_indices.append((start_idx, end_idx))
#                     literal_extra_texts.append((before_text, after_text))

#             with time_block(f"input_{i}_fetch_partition_input_ids", timers):
#                 input_ids = get_sub_sqls(template_tokens, sql_literal_indices)
#                 sql_literal_types = [token.token_type for token in sql_literal_tokens]
#                 query_token_stats[0] = sum([len(i) for i in input_ids])
#                 # process prmopt tokens before the first literal
#                 # outputs = model(llm_input)
#                 # past_key_values = outputs["past_key_values"]
#                 past_key_values = None
#                 model_kwargs = {}

#             # initialize 
#             for idx, j in enumerate(sql_literal_types):
#                 with time_block(f"literal_{i}_regex_init", timers):
#                     if j == TokenType.NUMBER:
#                         query_token_stats[4] += 1
#                         next_token_id = ""
#                         prev_token_id = "" # prev token id is used to simulate token healing which will happen naturally with GCD
                        
#                         if token_healing:
#                             if len(input_ids[idx+1]) > 0:
#                                 next_token_id = input_ids[idx+1].pop(0)
#                             if len(input_ids[idx]) > 0:
#                                 prev_token_id = input_ids[idx].pop()
#                         elif token_healing_right:
#                             if len(input_ids[idx+1]) > 0:
#                                 next_token_id = input_ids[idx+1].pop(0) #rhs token healing

#                         regex = f"""{re.escape(literal_extra_texts[idx][0])}{sql_number_rule}{re.escape(literal_extra_texts[idx][1])}"""

#                         regex = f"""{re.escape(tokenizer.decode(prev_token_id)) if prev_token_id != "" else ""}{regex}{re.escape(tokenizer.decode(next_token_id)) if next_token_id != "" else ""}"""
                        
#                     else:
#                         query_token_stats[5] += 1
#                         next_token_id = ""
#                         prev_token_id = "" # prev token id is used to simulate token healing which will happen naturally with GCD
                        
#                         if token_healing:
#                             if len(input_ids[idx+1]) > 0:
#                                 next_token_id = input_ids[idx+1].pop(0)
#                             if len(input_ids[idx]) > 0:
#                                 prev_token_id = input_ids[idx].pop()
#                         elif token_healing_right:
#                             if len(input_ids[idx+1]) > 0:
#                                 next_token_id = input_ids[idx+1].pop(0) #rhs token healing

#                         regex = f"""{re.escape(literal_extra_texts[idx][0])}{string_rule}{re.escape(literal_extra_texts[idx][1])}"""

#                         regex = f"""{re.escape(tokenizer.decode(prev_token_id)) if prev_token_id != "" else ""}{regex}{re.escape(tokenizer.decode(next_token_id)) if next_token_id != "" else ""}"""

#                     query_token_stats[6] += 1
                
#                 with time_block(f"literal_{i}_processor_creation", timers):
#                     # regex processor
#                     processor = outlines.processors.RegexLogitsProcessor(
#                                     regex,
#                                     outlines_tokenizer,
#                                 )
#                     print(f"Processor {i} created with regex: {regex}")
#                     # current_processors[idx] = (processor, regex, j)
#                     current_processors[idx] = processor
#                     logit_processors.append(current_processors[idx])
#                 with time_block(f"literal_{i}_kv_cache_update", timers):
#                     if past_key_values is not None:
#                         model_kwargs["past_key_values"] = past_key_values
                
#                 with time_block(f"literal_{i}_retokenization", timers):
#                     # add new tokens to the input_ids
#                     if len(input_ids[idx]) > 0:
#                         if TOKENIZER_GUIDED:    
#                             input_ids_str = tokenizer.decode(torch.tensor(input_ids[idx]), skip_special_tokens=True)
#                             retokenized = tokenizer(input_ids_str, return_tensors="pt")["input_ids"].to(device)
#                             if "llama" in model_key and retokenized[0][0] == tokenizer.bos_token_id:
#                                 retokenized = retokenized[:, 1:]
#                             llm_input = torch.cat((llm_input, retokenized), dim=1)
#                             # remove_beginning_bos_tokens(llm_input, tokenizer.bos_token_id)
#                         else:
#                             llm_input = torch.cat((llm_input, torch.tensor(input_ids[idx]).view(1, -1).to(device)), dim=1)
#                     else:
#                         # handles the case where the input_ids is empty so we need to ensure kv cache has 1 seq less than the input_ids otherwise transformers library will throw an error
#                         past_key_values = remove_trailing_kv_cache(past_key_values, model.__class__.__name__)
#                         model_kwargs["past_key_values"] = past_key_values
                    
#             #     with time_block(f"literal_{i}_generate_call", timers):
#             #         outputs = model.generate(llm_input, 
#             #                                 pad_token_id=tokenizer.eos_token_id, 
#             #                                 max_new_tokens=100, 
#             #                                 logits_processor=[processor],
#             #                                 return_dict_in_generate=True,
#             #                                 output_scores=True,
#             #                                 output_logits=True,
#             #                                 output_hidden_states=True,
#             #                                 do_sample=False,
#             #                                 **model_kwargs
#             #                                 )

#             #     with time_block(f"literal_{i}_decoding", timers):
#             #         # Remove only trailing `eos_token_id`
#             #         output_token_ids = remove_trailing_eos_tensor(outputs["sequences"][0, llm_input.shape[1]:], tokenizer.eos_token_id)

#             #         literal_counter += len(output_token_ids)
#             #         if j == TokenType.NUMBER:
#             #             query_token_stats[1] += len(output_token_ids)
#             #         else:
#             #             query_token_stats[2] += len(output_token_ids)
                    
#             #         query_token_stats[3] += len(output_token_ids)

#             #     with time_block(f"literal_{i}_extract_kv_cache", timers):
#             #         # kv caching
#             #         past_key_values = truncate_kv_cache(
#             #             outputs["past_key_values"], 
#             #             llm_input.shape[1] + len(output_token_ids),
#             #             model.__class__.__name__
#             #         )

#             #     with time_block(f"literal_{i}_concatenation", timers):
#             #         # add output token ids to the llm_input
#             #         llm_input = torch.cat((llm_input, output_token_ids.view(1, -1).to(device)), dim=1)

#             # with time_block(f"literal_{i}_final_processing", timers):
#             #     if  len(input_ids[-1]) > 0:
#             #         llm_input = torch.cat((llm_input, torch.tensor(input_ids[-1]).view(1, -1).to(device)), dim=1)
                    
#             #     output_token_ids = llm_input[0][prompt_token_count:]
#             #     token_counter += len(output_token_ids)
#             #     out_ids.update({str(i): str(output_token_ids.tolist())})
#             #     sql_text = tokenizer.decode(output_token_ids, skip_special_tokens=True)
#             #     responses.update({str(i): sql_text})
#             #     token_stats.append(query_token_stats)
#         except Exception as e:
#             import traceback
#             errors.append(f"{i}: {e}\n{traceback.format_exc()}")
#             responses.update({str(i): ""})
#             out_ids.update({str(i): ""})

#     end_event3.record()
#     torch.cuda.synchronize()
#     elapsed_time3 = start_event3.elapsed_time(end_event3)
#     model.e3 += elapsed_time3

#     with open(f'{TEMPLATE_PREDS}/{dataset}/{model_key}/{method_output_path}/{variant}/logit_processors.pkl', "wb") as f:
#         pickle.dump(logit_processors, f)

#     # # deeper analysis for latency
#     # # Extract unique operations (without iteration numbers)
#     # operations = set()
#     # iteration_data = {}
    
#     # for timer_name, time_ms in timers.items():
#     #     # Parse timer name to extract iteration and operation
#     #     parts = timer_name.split('_')
        
#     #     if len(parts) >= 2 and parts[1].isdigit():
#     #         iteration = int(parts[1])
#     #         operation = '_'.join(parts[2:])  # Everything after the iteration number
#     #     else:
#     #         # Handle cases without iteration numbers
#     #         iteration = 0
#     #         operation = timer_name
        
#     #     operations.add(operation)
        
#     #     if iteration not in iteration_data:
#     #         iteration_data[iteration] = {}
        
#     #     iteration_data[iteration][operation] = time_ms
    
#     # # Convert to DataFrame
#     # operations = list(operations)
#     # iterations = sorted(list(iteration_data.keys()))
    
#     # # Create DataFrame with iterations as rows and operations as columns
#     # timing_matrix = []
#     # for iteration in iterations:
#     #     row = []
#     #     for operation in operations:
#     #         time_value = iteration_data[iteration].get(operation, np.nan)
#     #         row.append(time_value)
#     #     timing_matrix.append(row)
    
#     # df = pd.DataFrame(timing_matrix, index=iterations, columns=operations)
    
#     # # Add summary statistics
#     # df_with_stats = df.copy()

#     # token_stats_df = pd.DataFrame(token_stats, columns=token_stats_columns)
#     # token_stats_df['TOTAL_PER_ITERATION'] = token_stats_df.sum(axis=1, skipna=True)
#     # token_stats_df.loc['TOTAL_PER_OPERATION'] = token_stats_df.sum(axis=0, skipna=True)

#     # # Add row totals (total time per iteration)
#     # df_with_stats['TOTAL_PER_ITERATION'] = df_with_stats.sum(axis=1, skipna=True)
#     # df_with_stats.loc['TOTAL_PER_OPERATION'] = df_with_stats.sum(axis=0, skipna=True)
    
#     # # Save the main timing table
    
#     # # save model kwargs
#     # if token_healing:
#     #     suffix = "_th"
#     # elif token_healing_right:
#     #     suffix = "_th_right"
#     # else:
#     #     suffix = ""

#     # # save responses
#     # timing_table_with_stats_path = f"results/{dataset}/{model_key}/{method_output_path}/{variant}/timing_table_stats{suffix}.csv"
#     # # timing_table_path = f"results/{dataset}/{model_key}/{method_output_path}/{variant}/timing_table{suffix}.csv"
#     # os.makedirs(os.path.dirname(timing_table_with_stats_path), exist_ok=True)
#     # # df_with_stats.to_csv(timing_table_with_stats_path, index=False)

#     # token_stats_df_path = f"results/{dataset}/{model_key}/{method_output_path}/{variant}/token_stats{suffix}.csv"
#     # os.makedirs(os.path.dirname(token_stats_df_path), exist_ok=True)
#     # # token_stats_df.to_csv(token_stats_df_path, index=False)
    

#     # output_path = Path(f'{TEMPLATE_PREDS}/{dataset}/{model_key}/{method_output_path}/{variant}/inference{suffix}.json')
#     # if not os.path.exists(output_path.parent):
#     #     os.makedirs(output_path.parent)
#     # with open(output_path, 'w') as f:
#     #     json.dump(responses, f, indent=4)
#     # with open(Path(f'{TEMPLATE_PREDS}/{dataset}/{model_key}/{method_output_path}/{variant}/inference_tokens{suffix}.json'), 'w') as f:
#     #     json.dump(out_ids, f, indent=4)

#     # # save errors in text file
#     # with open(f'{TEMPLATE_PREDS}/{dataset}/{model_key}/{method_output_path}/{variant}/errors{suffix}.txt', 'w') as f:
#     #     for error in errors:
#     #         f.write(f"{error}\n")
    
#     # logging.info(f"File: {output_path}")
#     # logging.info(f"generate {model.e1}")
#     # logging.info(f"logit_processor {model.e2}")
#     # logging.info(f"for loop {model.e3}")
#     # logging.info(f"token count {token_counter}")
#     # logging.info(f"literal count {literal_counter}")

#     # from extras.ex_match import get_accuracy
    
#     # # get accuracy ex match
#     # db_path = os.getenv("BIRD_DEV_PATH") if dataset == "bird" else os.getenv("SPIDER_DEV_PATH")
#     # gold_query_path = BIRD_DEV_QUERIES_PATH if dataset == "bird" else SPIDER_DEV_QUERIES_PATH
#     # num_correct, total, result = get_accuracy(str(output_path), str(gold_query_path), str(db_path), dataset=dataset)
#     # accuracy = num_correct / total
    
#     # # save results
#     # results_path = f"results/{dataset}/{model_key}/{method_output_path}/{variant}/results{suffix}.pkl"
#     # os.makedirs(os.path.dirname(results_path), exist_ok=True)
#     # with open(results_path, 'wb') as pkl_file:
#     #     pickle.dump({"num_correct": num_correct, "total": total, "accuracy": accuracy, "token_count": token_counter, "literal_count": literal_counter, "total_time": model.e3}, pkl_file)
#     # with open(f"results/{dataset}/{model_key}/{method_output_path}/{variant}/ex_match{suffix}.txt", 'w') as file:
#     #     file.write(str(result))

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_id", type=str, default="codes_1b_bird_with_evidence")
#     parser.add_argument("--device", type=int, default=0) # pass using CUDA_VISIBLE_DEVICES
#     parser.add_argument("--method", type=str, choices=["tf_flexible", "tf_tight", "flexible_tokenizer_guided", "tight_tokenizer_guided"], default="tf_flexible")
#     parser.add_argument("--token_healing", action='store_true', default=False)
#     parser.add_argument("--token_healing_right", action='store_true', default=False)
#     parser.add_argument("--dataset", type=str, choices=["bird", "spider"], default="bird")
#     parser.add_argument("--variant", type=str, choices=["main", "extended_ws", "extended_ws_plus", "extended_str_literal", "extended_ws_and_str_literal"])
#     args = parser.parse_args()

#     assert (args.token_healing and args.token_healing_right) == False, "Both token healing and token healing right cannot be true simultaneously"

#     log_file_path = f'logs/teacher_forcing_inference/{args.dataset}/{args.model_id}/{args.method}/{args.variant}/{TODAYS_DATE}_{TIMENOW}.log'
#     os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
#     logging.basicConfig(
#         filename=log_file_path, 
#         format='%(levelname)s:%(message)s',
#         filemode='w',
#         level=logging.INFO
#     )

#     logging.info(f"Arguments: {args}")

#     main(
#         args.model_id,
#         torch.device("cuda:{}".format(args.device)),
#         args.method,
#         args.token_healing,
#         args.token_healing_right,
#         args.dataset,
#         args.variant
#    )

