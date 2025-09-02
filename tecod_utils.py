import os
import sys
from dotenv import load_dotenv

# Load the .env file
# load_dotenv()

# sys.path.insert(1, os.getenv("PROJECT_ROOT"))
# os.environ['HF_HOME'] = os.getenv("HF_CACHE")

import sqlglot
from sqlglot import parse_one, exp
from sqlglot.expressions import Select, Table, Column, Alias, CTE, Subquery
from sqlglot.tokens import Token, TokenType
import copy
from .constants import *
import json
from pathlib import Path
import re
import torch



def get_tables_and_columns_from_db(db_name: str = None, dataset='bird', db_path: str = None) -> tuple:
    """
    Get the tables and columns from the database connection.
    Args:
    ----
    db_name: str
        Database name
    dataset: str
        Dataset type ('bird' or 'spider')
    db_path: str
        Direct path to database file. If provided, takes precedence over dataset-based path construction.

    Returns:
    ----
    tables: list
        List of table names in the database
    tab_set_lower: dict
        Dictionary of table names in lower case
    col_set_lower: dict
        Dictionary of column names in lower case

    """
    import sqlite3

    assert db_path is not None or (db_name is not None and dataset is not None), "Either db_path must be provided or both db_name and dataset must be provided"

    if db_path is None:
        if dataset == 'bird':
            db_path = os.path.join(os.getenv("BIRD_DEV_PATH"), "dev", "dev_databases", db_name, f'{db_name}.sqlite')
        elif dataset == 'spider':
            db_path = os.path.join(os.getenv("SPIDER_DEV_PATH"), "database", db_name, f'{db_name}.sqlite')
    
    db_conn = sqlite3.connect(db_path)

    # global tables, tab_set_lower, col_set_lower

    if db_conn is None:
        raise ValueError("Database connection is None")

    # Get the list of tables in the database
    tables = db_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    tables = [table[0] for table in tables]
    columns_names = []

    # add backticks if needed to the table names
    # tables = [add_back_ticks(table) for table in tables]

    # Get the columns for each table
    tab_set = {}
    col_set = {}
    for table in tables:
        try:
            columns = db_conn.execute(f"PRAGMA table_info({table})").fetchall()
        except sqlite3.OperationalError as e:
            columns = db_conn.execute(f"PRAGMA table_info(`{table}`)").fetchall()
        col_names = [col[1] for col in columns]
        tab_set[table.lower()] = table
        col_set[table.lower()] = col_names
        # columns_names.extend([add_back_ticks(col) for col in col_names])
        columns_names.extend([col for col in col_names])

    return tables, columns_names, tab_set, col_set

def extract_sqlite_aliases(query: str) -> dict:
    """
    Extract all aliases and their corresponding values from a SQLite query.
    
    Args:
        query: The SQLite query to parse
        
    Returns:
        A dictionary mapping aliases to their corresponding values and types
    """
    try:
        # Parse the query using SQLite dialect
        parsed = parse_one(query, dialect='sqlite')
        if not parsed:
            return {}
    except Exception as e:
        print(f"Error parsing query: {e}")
        return {}
    
    aliases = {}
    
    # Process all nodes in the AST
    for node in parsed.walk():
        # Column aliases
        if isinstance(node, Alias):
            try:
                aliases[node.alias] = {
                    'value': node.this.sql(dialect='sqlite'),
                    'type': 'column'
                }
            except Exception as e:
                print(f"Error processing column alias {getattr(node, 'alias', 'unknown')}: {e}")
        
        # Table aliases
        elif isinstance(node, Table) and hasattr(node, 'alias') and node.alias:
            try:
                aliases[node.alias] = {
                    'value': node.name,
                    'type': 'table'
                }
            except Exception as e:
                print(f"Error processing table alias {getattr(node, 'alias', 'unknown')}: {e}")
        
        # Subquery aliases
        elif isinstance(node, Subquery) and hasattr(node, 'alias') and node.alias:
            try:
                aliases[node.alias] = {
                    'value': f"({node.this.sql(dialect='sqlite')})",
                    'type': 'subquery'
                }
            except Exception as e:
                print(f"Error processing subquery alias {getattr(node, 'alias', 'unknown')}: {e}")
        
        # CTE aliases
        elif isinstance(node, CTE):
            try:
                aliases[node.alias] = {
                    'value': node.this.sql(dialect='sqlite'),
                    'type': 'cte'
                }
            except Exception as e:
                print(f"Error processing CTE alias {getattr(node, 'alias', 'unknown')}: {e}")
    
    return aliases


def convert_sql_string_to_template(sql_string, db_name=None, dataset_name=None, mask_literals=True, db_path=None) -> str:
    """
    use sqlglot transform to replace string literals with "string_rule" and number_literals with "number_rule"
    """
    tree = parse_one(sql_string, read="sqlite")

    tables, columns, _, _ = get_tables_and_columns_from_db(db_name=db_name, dataset=dataset_name, db_path=db_path)
    aliases_dict = extract_sqlite_aliases(sql_string)
    aliases_names = list(aliases_dict.keys())

    def transform_tree(node, tables, columns, aliases):
        if isinstance(node, exp.Identifier):
            if node.name not in tables and node.name not in columns and node.name not in aliases:
                return exp.Literal.string("string_rule")
        if isinstance(node, exp.Literal):
            if node.is_string:
                return exp.Literal.string("string_rule")
            else:
                return exp.Literal.number("number_rule")
        return node

    if mask_literals:
        tree = tree.transform(lambda node: transform_tree(node, tables=tables, columns=columns, aliases=aliases_names))    
    sql_template = tree.sql(dialect='sqlite')
    sql_template = sql_template.replace("'string_rule'", "string_rule")

    return sql_template


def remove_aliases_from_sql(sql):
    """
    Remove aliases from the SQL query.
    Args:
        sql: The SQL query string
    Returns:
        The SQL query string without aliases
    """
    parsed_query = parse_one(sql, read="sqlite")

    aliases_dict = extract_sqlite_aliases(sql)

    def remove_alias_mentions(node):
        if isinstance(node, exp.Table):
            node.set('alias', None)
            return node

        if isinstance(node, exp.Alias):
            return node.this
        
        return node

    parsed_query = parsed_query.transform(remove_alias_mentions)

    def substitute_alias(node, aliases):
        if isinstance(node, exp.Column): # handle T1.column
            if node.table in list(aliases.keys()):
                if aliases[node.table]['type'] == 'table':
                    node.set('table', aliases[node.table]['value'])

        if isinstance(node, exp.Identifier):
            if node.name in list(aliases.keys()):
                if aliases[node.name]['type'] in ['column', 'table']:
                    node.set('this', aliases[node.name]['value'])

        return node

    parsed_query = parsed_query.transform(lambda node: substitute_alias(node, aliases_dict))

    return parsed_query.sql(dialect='sqlite')


def convert_template_to_ebnf(template, remove_aliases=False, db_id=None, dataset=None, type=None, db_path=None) -> tuple[str, dict]:
    """
    Generates an EBNF string from an SQL query.
    Args:
    ----
    query: str
        SQL template string to convert to EBNF

    remove_aliases: bool
        If True, removes aliases from the query

    db_id: str
        Database ID to fetch tables and columns from

    dataset: str
        Dataset name to fetch tables and columns from

    type: str
        Type of grammar to generate (add support for different grammars here if required). Options are:
        - 'base_grammar'
        - 'tab_and_col_to_rule_grammar'
        - 'common_table_and_column_rule_grammar'

    db_path: str
        Direct path to database file. If provided, takes precedence over dataset-based path construction.

    Returns:
    masked_query: str
        EBNF string

    db_id: str
        Database ID to fetch tables and columns from

    dataset: str
        Dataset name to fetch tables and columns from

    type: str
        Type of grammar to generate (add support for different grammars here if required). Options are:
        - 'base_grammar'
        - 'tab_and_col_to_rule_grammar'
        - 'common_table_and_column_rule_grammar'

    Returns:
    masked_query: str
        EBNF string

    Example:
    --------
    >>> convert_template_to_ebnf("SELECT * FROM table WHERE column = 123", db_id="california_schools", dataset="bird", type="base_grammar")
    'select ws "*" ws from ws "table" ws where ws "column" ws "=" ws number_rule'

    Author:
    --------
    - @sarvam31
    """
    # remove alias from the query
    # query_ = None

    assert db_path is not None or (db_id is not None and dataset is not None), "Either db_path must be provided or both db_id and dataset must be provided"
    assert type in ['base_grammar', 
                    'tab_and_col_to_rule_grammar',
                    'common_table_and_column_rule_grammar'], "Grammar must be as mentioned in the docstring"

    query = template.strip()
    if remove_aliases:
        query = remove_aliases_from_sql(query)

    new_rules_for_ebnf = {}
    rules_in_template = ["number_rule", "string_rule"]
    add_back_ticks = lambda x: f"`{x}`" if " " in x else x

    if type == 'base_grammar':
        pass

    elif type == 'tab_and_col_to_rule_grammar':
        aliases_dict = extract_sqlite_aliases(query)
        tables, columns, tab_set, col_set = get_tables_and_columns_from_db(db_name=db_id, dataset=dataset, db_path=db_path)

        # find tables and columns present in query
        tables_present = []
        columns_present = []

        for idx,c in enumerate(columns):
            if c.lower().strip('`') in query.lower():
                columns_present.append(c)
            
        for idx, t in enumerate(tables):
            if t.lower().strip('`') in query.lower():
                tables_present.append(t)

        alias_to_rule_name = {}
        table_to_rule_name = {}

        parsed_query = parse_one(query, dialect='sqlite')
        if not parsed_query:
            return None, {}

        def transform(node, value, rule, aliases_dict=None, alias_to_rule_name=None, table_to_rule_name=None):
            # Replace table names
            if isinstance(node, exp.Table) and node.name.lower() == value.lower().strip('`'):
                if node.alias:
                    return exp.Table(this=exp.to_identifier(rule), alias=exp.TableAlias(this=exp.to_identifier(node.alias)))
                return exp.Table(this=exp.to_identifier(rule))

            if isinstance(node, exp.TableAlias) and node.name.lower() == value.lower().strip('`'):
                return exp.Alias(this=exp.to_identifier(rule))
            
            if isinstance(node, exp.Column) and node.name.lower() == value.lower().strip('`'):
                if node.table:
                    if node.table in aliases_dict.keys():
                        return exp.Column(this=exp.to_identifier(rule), table=exp.to_identifier(alias_to_rule_name[node.table]))
                    #####
                    # if alias, then just add new entry, special case: partial expression 
                    try:
                        return exp.Column(this=exp.to_identifier(rule), table=exp.to_identifier(table_to_rule_name[node.table]))
                    except KeyError:
                        rule_name = f"alias_rule_{len(aliases_dict)}"
                        a = add_back_ticks(node.table)
                        values = f"{a.lower()}|{a.upper()}|{a.title()}"
                        new_rules_for_ebnf[rule_name] = values
                        alias_to_rule_name[a] = rule_name
                        try:
                            return exp.Column(this=exp.to_identifier(rule), table=exp.to_identifier(alias_to_rule_name[add_back_ticks(node.table)]))
                        except KeyError as e:
                            raise e
                    #####
                return exp.Column(this=exp.to_identifier(rule))

            if isinstance(node, exp.Alias) and node.alias.lower() == value.lower().strip('`'):
                return exp.Alias(this=node.this, alias=exp.to_identifier(alias_to_rule_name[value]))

            return node

        # create new rules for alias separately to generate alias_to_rule_name mapping before any transformations
        for idx, a in enumerate(aliases_dict): 
            rule_name = f"alias_rule_{idx}"
            a = add_back_ticks(a)
            values = f"{a.lower()}|{a.upper()}|{a.title()}"
            new_rules_for_ebnf[rule_name] = values
            alias_to_rule_name[a] = rule_name

        for idx, t in enumerate(tables_present):
            rule_name = f"table_rule_{idx}"
            t = add_back_ticks(t)
            values = f"{t.lower()}|{t.upper()}|{t.title()}"
            new_rules_for_ebnf[rule_name] = values
            table_to_rule_name[t] = rule_name

        # create new rules for tables, columns and aliases
        for idx, t in enumerate(tables_present):
            # remove table name from the query using parsed_query
            parsed_query = parsed_query.transform(lambda node: transform(node, t, table_to_rule_name[t], aliases_dict, alias_to_rule_name, table_to_rule_name))

        for idx, a in enumerate(aliases_dict): 
            rule_name = alias_to_rule_name[a]
            parsed_query = parsed_query.transform(lambda node: transform(node, a, rule_name, aliases_dict, alias_to_rule_name, table_to_rule_name))

        for idx, c in enumerate(columns_present):
            rule_name = f"column_rule_{idx}"
            c = add_back_ticks(c)
            values = f"{c.lower()}|{c.upper()}|{c.title()}"
            new_rules_for_ebnf[rule_name] = values
            parsed_query = parsed_query.transform(lambda node: transform(node, c, rule_name, aliases_dict, alias_to_rule_name, table_to_rule_name))

        # generate query from parsed_query
        query = parsed_query.sql(dialect='sqlite')
        rules_in_template.extend(list(new_rules_for_ebnf.keys()))

    # Tokenize the SQL query
    expression = sqlglot.tokenize(query, read="sqlite")
    literal_quote = '\\"'
    optional_ws = " ws? "

    def templatise_node(node):
        templatised_node = copy.deepcopy(node)
        
        if node.text in rules_in_template:
            return templatised_node
        # if mask_literals:
        #     if node.text in new_rules:
        #         return templatised_node
        
        if node.token_type in [TokenType.STRING, TokenType.NUMBER]:
                if query[node.start] in ["'", '"', "`"]:
                    templatised_node.text = '"' + query[node.start] + node.text.replace("'", "''") + query[node.end] + '"'
                else:
                    templatised_node.text = '"' + node.text + '"'
                return templatised_node

        elif node.token_type in [
            TokenType.IDENTIFIER,
            TokenType.VAR,
            TokenType.TABLE,
            TokenType.COLUMN,
        ]:
            if query[node.start] in ["'", '"', "`"]:
                # Add literal_quote to the start and end of the node text to make it part of output template
                templatised_node.text = (
                    # '"' + query[node.start] + node.text + query[node.end] + '"'
                    '"' + literal_quote + node.text + literal_quote + '"'
                )
            else:
                templatised_node.text = '"' + node.text + '"'
        elif (
            node.text.lower().replace(" ", "_") in SQL_KEYWORDS
            or node.text.lower() in SQL_FUNCTIONS
            or node.text.lower() in SQL_DATATYPES
        ):
            templatised_node.text = node.text.upper().replace(" ", "_")
        elif node.text in OPERATORS or node.text in SYMBOLS:
            templatised_node.text = '"' + node.text + '"'
        else:
            templatised_node.text = '"' + node.text.upper() + '"'

        return templatised_node

    query_tokens = list(map(templatise_node, expression))

    def join_tokens(tokens):
        """
        Join the tokens with a space and return the string.
        """
        collected_tokens = []
        for idx, tok in enumerate(tokens):
            if (tok.text.strip("\"") in OPERATORS or tok.text.strip("\"") in SYMBOLS) and (tok.text.strip("\"") != "."):
                collected_tokens[-1] = "WS?"
                collected_tokens.append(tok.text)
                if idx < len(tokens) - 1:
                    collected_tokens.append("WS?")
                    # not sure why this is needed, but if some error cases are found consider uncommenting this
                    # if tok.text.strip("\"") == ")":
                    #     collected_tokens.append("WS")
                    # else:
                    #     collected_tokens.append("WS?")
            else:
                collected_tokens.append(tok.text)
                if idx < len(tokens) - 1:
                    collected_tokens.append("WS")
        return " ".join(collected_tokens).replace(" WS \".\" WS ", "\".\"")
    return join_tokens(query_tokens), new_rules_for_ebnf


def ebnf_to_regex(ebnf_str, full_sql=True, new_rules_for_ebnf=None, **kwargs):
    # DOUBLE_QUOTE_REGEX = "|\"([^\"\\\\]|\\\\.)*\""
    # EXTENDED_WS = '[ \u000B\t\r\n]'
    # # EXTENDED_WS_PLUS = '[ \u000B\t\r\n]+'
    # EXTENDED_WS_PLUS = '\\s|\\s\\s'
    if full_sql:
        regex_grammar = "WS?{select_stmt}(WS?|SEMICOLON?)".format(select_stmt=ebnf_str)
    else:
        regex_grammar = ebnf_str
    
    if 'regex_rules' in kwargs:
        regex_rules = kwargs['regex_rules']
    elif Path('complete_sql_template.json').exists():
        regex_rules = json.loads(open('complete_sql_template.json', "r").read())
    elif (Path(os.environ['ROOT_DIR']) / 'src/pdec/complete_sql_template.json').exists():
        regex_rules = json.loads(open(Path(os.environ['ROOT_DIR']) / 'src/pdec/complete_sql_template.json', "r").read())
    else:
        regex_rules = json.loads(open(GRAMMAR_TEMPLATE_JSON_PATH, "r").read())
    
    if new_rules_for_ebnf is not None:
        for key, value in new_rules_for_ebnf.items():
            if key not in regex_rules:
                regex_rules[key] = value

    
    double_quoted_substrings = re.findall(r'(?<!\\)"(.*?)(?<!\\)"', regex_grammar)
    for dq in double_quoted_substrings:
        dq_ = dq.replace("\\\"", "\"")
        regex_grammar = regex_grammar.replace(f'"{dq}"', re.escape(dq_))
    
    sorted_keys = sorted(regex_rules.keys(), key=len, reverse=True)
    # Initialize the result string
    result = ""
    remaining = regex_grammar
    
    while remaining:
        replaced = False
        for key in sorted_keys:
            match = re.match(re.escape(key), remaining)
            if match:
                if key not in ["DOUBLEQUOTE", "SINGLEQUOTE", "string_rule", "number_rule", "WS","SEMICOLON", "projections", "conditions"]:
                    result += "(" + "|".join([re.escape(i) for i in regex_rules[key].split("|")]) + ")"
                else:
                    result += "(" + regex_rules[key] + ")"
                remaining = remaining[match.end():]
                replaced = True
                break
        if not replaced:
            result += remaining[0]
            remaining = remaining[1:]

    return result.replace(f" ({regex_rules["WS"]}) ", "(" + regex_rules["WS"] + ")").replace(f" ({regex_rules["WS"]})? ", "(" + regex_rules["WS"] + ")?")


def remove_trailing_kv_cache(past_key_values, model_type, t=-1):
    """
    Truncate past key-values to remove the last token.
    This is necessary when generating sequences token-by-token and # entries in kv_cache = # entries in input_ids
    """
    # if "Llama" in model_type or "Granite" in model_type:
        # Llama: [num_layers][2][batch, num_heads, seq, head_dim]
    return past_key_values.crop(max_length=t)
        
    
def truncate_kv_cache(past_key_values, new_seq_len, model_type):
    """Fast KV cache truncation"""
    # if "Llama" in model_type or "Granite" in model_type:
        # Llama: List[num_hidden_layers-32][2][batch_size, kv heads, seq_len, 128]
        # Dynamic cache: [num_hidden_layers][2(for key and values)][batch_size, num_heads, seq_len, head_dim]
        # Avoid list comprehension for speed
    past_key_values.crop(max_length=new_seq_len)
    return past_key_values
        
def remove_trailing_eos_tensor(tensor: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """Fast right-to-left scan for trailing EOS tokens"""
    # Get indices where tokens don't match EOS
    non_eos_positions = (tensor != eos_token_id).nonzero()
    
    # If empty tensor or all EOS, return empty
    if len(non_eos_positions) == 0:
        return tensor[:0]
        
    # Get rightmost non-EOS position
    last_non_eos = non_eos_positions[-1].item()
    
    # Return slice up to and including last non-EOS
    return tensor[:last_non_eos + 1]


def get_token_offsets(tokenizer, token_ids: torch.Tensor) -> list[tuple[int, int]]:
    """
    Get character offset mappings for a sequence of token IDs.
    
    Args:
        tokenizer: The tokenizer used to decode tokens
        token_ids: Tensor of token IDs
        
    Returns:
        List of tuples containing (start, end) character offsets for each token
    """
    # Decode tokens to get the full text
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    offsets = []
    current_pos = 0
    collected_tokens = []
    
    # Decode each token individually to get its text
    for token_id in token_ids:
        # Skip special tokens
        if token_id in tokenizer.all_special_ids:
            continue

        collected_tokens.append(token_id)
            
        # Get text for current token
        token_text = tokenizer.decode(collected_tokens, skip_special_tokens=True)
        
        # Find token text in decoded string starting from current position
        token_start = decoded_text.find(token_text, current_pos)
        if token_start == -1:
            # Handle case where token can't be found exactly (e.g. due to merged tokens)
            # token_start = current_pos
            # collected_tokens.append(token_id)
            continue
            
        token_end = token_start + len(token_text)
        for _ in range(0, len(collected_tokens)-1):
            offsets.append((token_end, token_end))
        offsets.append((token_start, token_end))
        collected_tokens = []
        
        current_pos = token_end
        
    return offsets

def get_covering_token_ids(tokenizer, token_ids: list, literal_span: tuple, decoded_tokens_with_spans: list) -> tuple[int, int]:
    """
    Find the token ID sequence that completely covers a given literal span.
    
    Args:
        tokenizer: HuggingFace tokenizer
        token_ids: List of token IDs
        literal_span: Tuple of (start, end) character positions for the literal
        decoded_tokens_with_spans: List of (token_text, (start, end)) tuples for each token
        
    Returns:
        Tuple of (start_token_idx, end_token_idx) that covers the literal
    """
    literal_start, literal_end = literal_span
    start_token_idx = None
    end_token_idx = None
    
    # Find starting token that contains or is before literal start
    for idx, (token_start, token_end) in enumerate(decoded_tokens_with_spans):
        if token_start <= literal_start < token_end:
            start_token_idx = idx
            break
            
    # Find ending token that contains or is after literal end
    for idx, (token_start, token_end) in enumerate(decoded_tokens_with_spans):
        if token_start <= literal_end < token_end:
            end_token_idx = idx + 1  # +1 to make it exclusive end index
            break
    
    if start_token_idx is None or end_token_idx is None:
        raise ValueError(f"Could not find tokens covering literal span {literal_span}")
        
    # Verify the coverage is complete
    covered_text = tokenizer.decode(token_ids[start_token_idx:end_token_idx])
    literal_text = tokenizer.decode(token_ids).encode('utf-8')[literal_start:literal_end+1].decode('utf-8')
    
    if literal_text not in covered_text:
        raise ValueError(f"Token sequence does not completely cover literal: '{literal_text}' not in '{covered_text}'")
    
    # find text outside of literal_text in covered_text for regex 
    start_idx = covered_text.find(literal_text) # find text present in covered_text but not in literal_text before start of literal_text
    end_idx = covered_text.find(literal_text) + len(literal_text) # find text present in covered_text but not in literal_text after end of literal_text   
    
    before_text = covered_text[:start_idx] if start_idx > 0 else ""
    after_text = covered_text[end_idx:] if end_idx < len(covered_text) else ""
        
    return start_token_idx, end_token_idx, before_text, after_text


def get_sub_sqls(template_tokens, sql_literal_indices):
    """
    Get the SQL template and literal indices for a sub-template.    "
    """
    # convert sql literal indices from list of tuples to 1d list
    sql_literal_indices = [item for sublist in sql_literal_indices for item in sublist]
    sql_literal_indices.insert(0, 0) # add the start of the template tokens
    sql_literal_indices.append(len(template_tokens)) # add the end of the template tokens

    sub_sql = []
    for i in range(0, len(sql_literal_indices), 2):
        sub_sql.append(template_tokens[sql_literal_indices[i]:sql_literal_indices[i+1]])

    return sub_sql

def decode_token_ids(tokenizer, model_output, inputs):
    """
    Decode the token ids to text , works for single input
    TODO: Add support for batch inputs
    """
    if "sequences" in model_output.keys():
        token_ids = model_output.sequences
    else:
        token_ids = model_output

    end_index = -1 if (token_ids[0].tolist()[-1] == tokenizer.eos_token_id) else None
    output_token_ids = token_ids[0].tolist()[inputs['input_ids'].shape[1]:end_index] # -1 for the eos token at the end
    
    return tokenizer.decode(output_token_ids, skip_special_tokens=True), output_token_ids

