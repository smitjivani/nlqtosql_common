DUMP_INTERVAL = 10 # Dump interval for saving predictions to disk

# Grammar for keywords and identifiers
GRAMMAR_TEMPLATE_EBNF_PATH = "generate_constrained_sqls/complete_sql_template.ebnf"
GRAMMAR_TEMPLATE_LARK_PATH = "generate_constrained_sqls/complete_sql_template.lark"
GRAMMAR_TEMPLATE_JSON_PATH = "generate_constrained_sqls/complete_sql_template.json"
GRAMMAR_TEMPLATE_ALTERNATE_EBNF_PATH = "generate_constrained_sqls/complete_sql_template_2.ebnf"

# not used in the current implementation
TEMP_RULE_MAPPING_PATH = "learn_distribution_parameters/temp_rule_mapping.pkl" 
# BAYESIAN_PARAMS_PATH = f'results/models/terminals_distribution_parameters/train_terminal_params.json'


TOKEN_ID_GRAMMAR_PATH = "learn_distribution_parameters/token_id_grammar.pkl" # not used in the current implementation
TOKEN_ID_PARAMS_PATH = f'results/models/terminals_distribution_parameters/dev_keyword_tok_seq_terminal_params.pkl'
IDENTIFIER_PARAMS_PATH = f'results/models/terminals_distribution_parameters/dev_identifier_tok_seq_terminal_params.pkl'

# was 3 before but changed to 1 to time it against gaed
NUM_SQL_PER_INPUT = 1 # Number of SQLs to generate per input

TERMINAL_DISTRIBUTION_PREDS = "data/preds/efficient_decoding_preds" # not used in the current implementation
ENUMERATION_PREDS = "data/preds" # not used in the current implementation
TEMPLATE_PREDS = "data/preds_tecod"

WS_RULE_TERMINALS = [' ', '  ', '   ', '    '] # not used in the current implementation

SQL_KEYWORDS = {
    'abort', 'action', 'add', 'after', 'all', 'alter', 'always', 'analyze', 'and', 'as', 'asc', 'attach', 'autoincrement', 'before', 'begin', 'between', 'cascade', 'case', 'cast', 'check', 'collate', 'column', 'commit', 'conflict', 'constraint', 'create', 'cross', 'current', 'current_date', 'current_time', 'current_timestamp', 'database', 'default', 'deferrable', 'deferred', 'delete', 'desc', 'detach', 'distinct', 'do', 'drop', 'each', 'else', 'end', 'escape', 'except', 'exclude', 'exclusive', 'exists', 'explain', 'fail', 'filter', 'first', 'following', 'for', 'foreign', 'from', 'full', 'generated', 'glob', 'group_by', 'groups', 'having', 'if', 'ignore', 'immediate', 'in', 'index', 'indexed', 'initially', 'inner', 'insert', 'instead', 'intersect', 'into', 'is', 'isnull', 'join', 'key', 'last', 'left', 'like', 'limit', 'match', 'materialized', 'natural', 'no', 'not', 'nothing', 'notnull', 'null', 'nulls', 'of', 'offset', 'on', 'or', 'order_by', 'others', 'outer', 'over', 'partition_by', 'plan', 'pragma', 'preceding', 'primary', 'query', 'raise', 'range', 'recursive', 'references', 'regexp', 'reindex', 'release', 'rename', 'replace', 'restrict', 'returning', 'right', 'rollback', 'row', 'rows', 'savepoint', 'select', 'set', 'table', 'temp', 'temporary', 'then', 'ties', 'to', 'transaction', 'trigger', 'unbounded', 'union', 'unique', 'update', 'using', 'vacuum', 'values', 'view', 'virtual', 'when', 'where', 'window', 'with', 'without'
}

# TODO: window functions are not included in the SQL_FUNCTIONS yet
SQL_FUNCTIONS = {
    'abs', 'changes', 'char', 'coalesce', 'concat', 'concat_ws', 'format', 'glob', 'hex', 'ifnull', 'iif', 'instr', 'last_insert_rowid', 'length', 'like', 'like', 'likelihood', 'likely', 'load_extension', 'load_extension', 'lower', 'ltrim', 'ltrim', 'max', 'min', 'nullif', 'octet_length', 'printf', 'quote', 'random', 'randomblob', 'replace', 'round', 'round', 'rtrim', 'rtrim', 'sign', 'soundex', 'sqlite_compileoption_get', 'sqlite_compileoption_used', 'sqlite_offset', 'sqlite_source_id', 'sqlite_version', 'substr', 'substr', 'substring', 'substring', 'total_changes', 'trim', 'trim', 'typeof', 'unhex', 'unhex', 'unicode', 'unlikely', 'upper', 'zeroblob', 'date', 'time', 'datetime', 'julianday', 'unixepoch', 'strftime', 'timediff', 'avg', 'count', 'count', 'group_concat', 'group_concat', 'max', 'min', 'string_agg', 'sum', 'total', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'ceil', 'ceiling', 'cos', 'cosh', 'degrees', 'exp', 'floor', 'ln', 'log', 'log', 'log10', 'log2', 'mod', 'pi', 'pow', 'power', 'radians', 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'trunc', 'json', 'jsonb', 'json_array', 'jsonb_array', 'json_array_length', 'json_array_length', 'json_error_position', 'json_extract', 'jsonb_extract', 'json_insert', 'jsonb_insert', 'json_object', 'jsonb_object', 'json_patch', 'jsonb_patch', 'json_pretty', 'json_remove', 'jsonb_remove', 'json_replace', 'jsonb_replace', 'json_set', 'jsonb_set', 'json_type', 'json_type', 'json_valid', 'json_valid', 'json_quote', 'json_group_array', 'jsonb_group_array', 'json_group_object', 'jsonb_group_object', 'json_each', 'json_each', 'json_tree', 'json_tree'
}

SQL_DATATYPES = {
    "real", "integer", "text", "true", "false", "float"
}

OPERATORS = {
    '=',
    '*',
    '+',
    '-',
    '~',
    '||',
    '/',
    '%',
    '<<',
    '>>',
    '&',
    '|',
    '<',
    '<=',
    '>',
    '>=',
    '==',
    '!=',
    '<>'
}

OPERATORS_TO_RULES_MAP = {
    "=": "equal",
    "*": "asterisk",
    "+": "plus",
    "-": "minus",
    "~": "tilde",
    "||": "concatenate",
    "/": "divide",
    "%": "modulo",
    "<<": "shift_left",
    ">>": "shift_right",
    "&": "bitwise_and",
    "|": "bitwise_or",
    "<": "less_than",
    "<=": "less_than_or_equal",
    ">": "greater_than",
    ">=": "greater_than_or_equal",
    "==": "double_equal",
    "!=": "not_equal",
    "<>": "alternate_not_equal",
}

SYMBOLS = {
    ';',
    '.',
    '(',
    ')',
    ',',
}

SYMBOLS_TO_RULES_MAP = {
    ";": "semicolon",
    ".": "dot",
    "(": "left_parenthesis",
    ")": "right_parenthesis",
    ",": "comma",
}

# get todays date in format DDMMYYYY
import datetime

TODAYS_DATE = datetime.datetime.now().strftime("%d%m%Y")
TIMENOW = datetime.datetime.now().strftime("%H%M%S")


#LLM constants and paths
from pathlib import Path

BIRD_TECOD_PROMPTS_PATH = "data/bird/dev/tecod/"
SPIDER_TECOD_PROMPTS_PATH = "data/spider/tecod/"

# codes 1b bird with evidence
CODES_1B_BIRD_EVIDENCE = "seeklhy/codes-1b-bird-with-evidence"
BIRD_CODES_1B_PROMPTS_PATH_0 = Path(BIRD_TECOD_PROMPTS_PATH) / f"{CODES_1B_BIRD_EVIDENCE.replace('/', '-')}_0.json"
BIRD_CODES_1B_PROMPTS_PATH_3 = Path(BIRD_TECOD_PROMPTS_PATH) / f"{CODES_1B_BIRD_EVIDENCE.replace('/', '-')}_3.json"

# codes 1b spider
CODES_1B_SPIDER = "seeklhy/codes-1b-spider"
SPIDER_CODES_1B_PROMPTS_PATH_0 = Path(SPIDER_TECOD_PROMPTS_PATH) / f"{CODES_1B_SPIDER.replace('/', '-')}_0.json"
SPIDER_CODES_1B_PROMPTS_PATH_3 = Path(SPIDER_TECOD_PROMPTS_PATH) / f"{CODES_1B_SPIDER.replace('/', '-')}_3.json"

# codes 15b bird with evidence
CODES_15B_BIRD_EVIDENCE = "seeklhy/codes-15b-bird-with-evidence"
BIRD_CODES_15B_PROMPTS_PATH_0 = Path(BIRD_TECOD_PROMPTS_PATH) / f"{CODES_15B_BIRD_EVIDENCE.replace('/', '-')}_0.json"
BIRD_CODES_15B_PROMPTS_PATH_3 = Path(BIRD_TECOD_PROMPTS_PATH) / f"{CODES_15B_BIRD_EVIDENCE.replace('/', '-')}_3.json"

# codes 15b spider
CODES_15B_SPIDER = "seeklhy/codes-15b-spider"
SPIDER_CODES_15B_PROMPTS_PATH_0 = Path(SPIDER_TECOD_PROMPTS_PATH) / f"{CODES_15B_SPIDER.replace('/', '-')}_0.json"
SPIDER_CODES_15B_PROMPTS_PATH_3 = Path(SPIDER_TECOD_PROMPTS_PATH) / f"{CODES_15B_SPIDER.replace('/', '-')}_3.json"

# llama 3.1 8b instruct
LLAMA_31_8B_INSTRUCT = "meta-llama/Llama-3.1-8B-Instruct"
BIRD_LLAMA_PROMPTS_PATH_0 = Path(BIRD_TECOD_PROMPTS_PATH) / f"{LLAMA_31_8B_INSTRUCT.replace('/', '-')}_0.json"
BIRD_LLAMA_PROMPTS_PATH_3 = Path(BIRD_TECOD_PROMPTS_PATH) / f"{LLAMA_31_8B_INSTRUCT.replace('/', '-')}_3.json"

SPIDER_LLAMA_PROMPTS_PATH_0 = Path(SPIDER_TECOD_PROMPTS_PATH) / f"{LLAMA_31_8B_INSTRUCT.replace('/', '-')}_0.json"
SPIDER_LLAMA_PROMPTS_PATH_3 = Path(SPIDER_TECOD_PROMPTS_PATH) / f"{LLAMA_31_8B_INSTRUCT.replace('/', '-')}_3.json"

# granite 3.1 2b instruct
GRANITE_31_2B_INSTRUCT = "ibm-granite/granite-3.1-2b-instruct"
BIRD_GRANITE_31_2BPROMPTS_PATH_0 = Path(BIRD_TECOD_PROMPTS_PATH) / f"{GRANITE_31_2B_INSTRUCT.replace('/', '-')}_0.json"
BIRD_GRANITE_31_2BPROMPTS_PATH_3 = Path(BIRD_TECOD_PROMPTS_PATH) / f"{GRANITE_31_2B_INSTRUCT.replace('/', '-')}_3.json"

SPIDER_GRANITE_31_2BPROMPTS_PATH_0 = Path(SPIDER_TECOD_PROMPTS_PATH) / f"{GRANITE_31_2B_INSTRUCT.replace('/', '-')}_0.json"
SPIDER_GRANITE_31_2BPROMPTS_PATH_3 = Path(SPIDER_TECOD_PROMPTS_PATH) / f"{GRANITE_31_2B_INSTRUCT.replace('/', '-')}_3.json"

# qwencoder 14b
XIYAN_SQL_QWENCODER_14B = "XGenerationLab/XiYanSQL-QwenCoder-14B-2504"
BIRD_QWENCODER_14B_PROMPTS_PATH_0 = Path(BIRD_TECOD_PROMPTS_PATH) / f"{XIYAN_SQL_QWENCODER_14B.replace('/', '-')}_0.json"
BIRD_QWENCODER_14B_PROMPTS_PATH_3 = Path(BIRD_TECOD_PROMPTS_PATH) / f"{XIYAN_SQL_QWENCODER_14B.replace('/', '-')}_3.json"

SPIDER_QWENCODER_14B_PROMPTS_PATH_0 = Path(SPIDER_TECOD_PROMPTS_PATH) / f"{XIYAN_SQL_QWENCODER_14B.replace('/', '-')}_0.json"
SPIDER_QWENCODER_14B_PROMPTS_PATH_3 = Path(SPIDER_TECOD_PROMPTS_PATH) / f"{XIYAN_SQL_QWENCODER_14B.replace('/', '-')}_3.json"

# qwencoder 7b
XIYAN_SQL_QWENCODER_7B = "XGenerationLab/XiYanSQL-QwenCoder-7B-2504"
BIRD_QWENCODER_7B_PROMPTS_PATH_0 = Path(BIRD_TECOD_PROMPTS_PATH) / f"{XIYAN_SQL_QWENCODER_7B.replace('/', '-')}_0.json"
BIRD_QWENCODER_7B_PROMPTS_PATH_3 = Path(BIRD_TECOD_PROMPTS_PATH) / f"{XIYAN_SQL_QWENCODER_7B.replace('/', '-')}_3.json"

SPIDER_QWENCODER_7B_PROMPTS_PATH_0 = Path(SPIDER_TECOD_PROMPTS_PATH) / f"{XIYAN_SQL_QWENCODER_7B.replace('/', '-')}_0.json"
SPIDER_QWENCODER_7B_PROMPTS_PATH_3 = Path(SPIDER_TECOD_PROMPTS_PATH) / f"{XIYAN_SQL_QWENCODER_7B.replace('/', '-')}_3.json"

# arctic text2sql r1
ARCTIC_TEXT2SQL_R1 = "Snowflake/Arctic-Text2SQL-R1-7B"
BIRD_ARCTIC_TEXT2SQL_R1_PROMPTS_PATH_0 = Path(BIRD_TECOD_PROMPTS_PATH) / f"{ARCTIC_TEXT2SQL_R1.replace('/', '-')}_0.json"
BIRD_ARCTIC_TEXT2SQL_R1_PROMPTS_PATH_3 = Path(BIRD_TECOD_PROMPTS_PATH) / f"{ARCTIC_TEXT2SQL_R1.replace('/', '-')}_3.json"

SPIDER_ARCTIC_TEXT2SQL_R1_PROMPTS_PATH_0 = Path(SPIDER_TECOD_PROMPTS_PATH) / f"{ARCTIC_TEXT2SQL_R1.replace('/', '-')}_0.json"
SPIDER_ARCTIC_TEXT2SQL_R1_PROMPTS_PATH_3 = Path(SPIDER_TECOD_PROMPTS_PATH) / f"{ARCTIC_TEXT2SQL_R1.replace('/', '-')}_3.json"

# GOLD QUERIES PATH
BIRD_DEV_QUERIES_PATH = "data/bird/dev/dev.json"
SPIDER_DEV_QUERIES_PATH = "data/spider/dev.json"
# BIRD_DEV_ICL_EXAMPLES_PATH = "data/bird/dev/icl_examples_bird.json"
# SPIDER_DEV_ICL_EXAMPLES_PATH = "data/spider/icl_examples_spider.json"


# llm mapping, refer these keys when passing model_id in scripts
model_id_mapping = {
    "codes_1b_bird_with_evidence": CODES_1B_BIRD_EVIDENCE,
    "codes_15b_bird_with_evidence": CODES_15B_BIRD_EVIDENCE,
    "llama_31_8b_instruct": LLAMA_31_8B_INSTRUCT,
    "granite_31_2b_instruct": GRANITE_31_2B_INSTRUCT,
    "codes_1b_spider": CODES_1B_SPIDER,
    "codes_15b_spider": CODES_15B_SPIDER,
    "xiyan_sql_qwencoder_14b": XIYAN_SQL_QWENCODER_14B,
    "xiyan_sql_qwencoder_7b": XIYAN_SQL_QWENCODER_7B,
    "arctic_text2sql_r1": ARCTIC_TEXT2SQL_R1
}

