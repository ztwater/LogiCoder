import os

class Globals:
    ROOT_DIR = '/root/DevEval'

    # ============================== Data ==============================
    BENCHMARK_METADATA_PATH = os.path.join(ROOT_DIR, 'data.jsonl')
    OUR_BENCHMARK_METADATA_PATH = os.path.join(ROOT_DIR, 'our_metadata.jsonl')
    FULL_BENCHMARK_METADATA_PATH = os.path.join(ROOT_DIR, 'full_metadata.jsonl')

    CODEREVAL_BENCHMARK_METADATA_PATH = os.path.join(ROOT_DIR, 'codereval_metadata.jsonl')
    # ==================================================================

    # ============================== Model ==============================
    LOCAL_MODEL_PATH = os.path.join(ROOT_DIR, 'models')
    UNIXCODER_PATH = os.path.join(LOCAL_MODEL_PATH, 'microsoft/unixcoder-base')
    # ===================================================================

    # ============================== Prompt ==============================
    PROMPT_DIR = os.path.join(ROOT_DIR, 'prompts')

    DOC_PROMPT_DIR = os.path.join(PROMPT_DIR, 'docs')
    JUDGE_PROMPT_DIR = os.path.join(PROMPT_DIR, 'judge')

    NO_RAG_PROMPT_DIR = os.path.join(PROMPT_DIR, 'no_rag')
    DEMO_DIR = os.path.join(PROMPT_DIR, 'demo')
    COT_PROMPT_DIR = os.path.join(PROMPT_DIR, 'cot')
    KA_PROMPT_DIR = os.path.join(PROMPT_DIR, 'knowledge_aware')
    REPOCODER_PROMPT_DIR = os.path.join(PROMPT_DIR, 'repocoder')
    LOGICODER_PROMPT_DIR = os.path.join(PROMPT_DIR, 'logicoder')
    # ====================================================================

    # ============================== Repo ==============================
    REPO_BASE_DIR = os.path.join(ROOT_DIR, 'Source_Code')
    REPO_BASE_CODEREVAL_DIR = os.path.join(ROOT_DIR, 'CoderEval')
    # ==================================================================

    # ============================== Knowledge Base ==============================
    KNOWLEDGE_BASE_DIR = os.path.join(ROOT_DIR, 'knowledge_base')
    WINDOW_DIR = os.path.join(KNOWLEDGE_BASE_DIR, 'window')
    REPO_WINDOW_DIR = os.path.join(WINDOW_DIR, 'repos')
    BENCHMARK_WINDOW_DIR = os.path.join(WINDOW_DIR, 'benchmark')
    GROUND_TRUTH_WINDOW_DIR = os.path.join(WINDOW_DIR, 'ground_truth')
    PREDICTION_WINDOW_DIR = os.path.join(WINDOW_DIR, 'prediction')
    REQ_WINDOW_DIR = os.path.join(WINDOW_DIR, 'req')
    COT_WINDOW_DIR = os.path.join(WINDOW_DIR, 'cot')
    DOC_KNOWLEDGE_DIR = os.path.join(WINDOW_DIR, 'docs')

    VECTOR_DIR = os.path.join(KNOWLEDGE_BASE_DIR, 'vector')
    RETRIEVAL_DIR = os.path.join(KNOWLEDGE_BASE_DIR, 'retrieval')

    VISITOR_DIR = os.path.join(KNOWLEDGE_BASE_DIR, 'visitor')
    UNIT_DIR = os.path.join(KNOWLEDGE_BASE_DIR, 'unit')
    CALL_GRAPH_DIR = os.path.join(KNOWLEDGE_BASE_DIR, 'call_graph')
    MAPPING_DIR = os.path.join(KNOWLEDGE_BASE_DIR, 'mapping')
    # CANDIDATE_DIR = os.path.join(KNOWLEDGE_BASE_DIR, 'candidate_func')
    CANDIDATE_DIR = UNIT_DIR
    # ============================================================================

    # ============================== Result ==============================
    PREDICTION_DIR = os.path.join(ROOT_DIR, 'predictions')
    # ====================================================================

    # ============================== CONFIG ==============================
    CONFIG_DIR = os.path.join(ROOT_DIR, 'config')
    LOG_CONFIG_FILE = os.path.join(CONFIG_DIR, 'logging.conf')
    API_KEY_FILE = os.path.join(CONFIG_DIR, 'api_keys.txt')
    # ====================================================================

