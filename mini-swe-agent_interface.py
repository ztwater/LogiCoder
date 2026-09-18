import json
import os
import shutil
import tempfile
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from minisweagent import package_dir
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import FormatError, Submitted
from minisweagent.models.litellm_model import LitellmModel

# ============================== Configuration ==============================
MODEL = "deepseeknew/deepseek-v4-pro"
API_KEYS_FILE = "/media/hd/RepoLevelGen/DevEval/config/api_keys_deepseek.txt"

PROMPT_FILE = "/media/hd/RepoLevelGen/DevEval/prompts/logicoder/logicoder-agent.jsonl"
OUTPUT_FILE = "./results/completion_mini-swe-agent_logicoder2_deepseek-v4-pro_T0_N1.jsonl"

REPO_BASE_DIR = "/media/hd/RepoLevelGen/DevEval/Source_Code"

MAX_WORKERS = 4
# ===========================================================================

# Load mini.yaml for its model/observation/format_error templates
_CODE_GEN_CONFIG = yaml.safe_load(Path("./codegen_config.yaml").read_text())

# # Custom system prompt: inject our task constraints, then append the mini.yaml
# # formatting instructions so the model knows how to issue bash tool calls.
# SYSTEM_TEMPLATE = (
#     "You are an expert Python developer. Your task is to implement the specified "
#     "function in the given repository. You can navigate the repository for related "
#     "context, but do NOT run or execute any tests, and do NOT add any code outside "
#     "the requested function.\n\n"
#     # + _CODE_GEN_CONFIG["agent"]["system_template"].strip()
# )
#
# # Instance template: {{task}} is rendered by DefaultAgent.run()
# INSTANCE_TEMPLATE = (
#     "{{task}}\n\n"
#     "When you are done implementing the function, submit by running:\n"
#     "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```\n"
#     "Do not combine that command with any other command."
# )


# ---------------------------------------------------------------------------
# Sandbox helpers
# ---------------------------------------------------------------------------

def setup_sandbox(entry: dict) -> str:
    """Copy the target repo to a temp dir and mask the function body with `pass`.

    Returns the sandbox repo root path. Caller must call cleanup_sandbox().
    """
    metadata = entry["metadata"]
    repo_path = metadata.get("repo_path", "")
    relative_path = metadata["relative_path"]
    body_start, body_end = metadata["body_position"]

    src_repo = os.path.join(REPO_BASE_DIR, repo_path)
    tmp_root = tempfile.mkdtemp(prefix="mini_swe_sandbox_")
    sandbox_repo = os.path.join(tmp_root, os.path.basename(src_repo))
    shutil.copytree(src_repo, sandbox_repo, symlinks=True)

    rel_in_repo = os.path.relpath(
        os.path.join(REPO_BASE_DIR, relative_path), src_repo
    )
    target_file = os.path.join(sandbox_repo, rel_in_repo)

    with open(target_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Detect indentation from the first body line; fall back to metadata value.
    indent = metadata.get("indent", 4)
    if body_start < len(lines):
        first_body_line = lines[body_start]
        stripped = first_body_line.lstrip()
        if stripped:
            indent = len(first_body_line) - len(stripped)

    masked = (
        lines[:body_start]
        + [" " * indent + "pass\n"]
        + lines[body_end:]
    )

    with open(target_file, "w", encoding="utf-8") as f:
        f.writelines(masked)

    return sandbox_repo


def cleanup_sandbox(sandbox_repo: str) -> None:
    tmp_root = os.path.dirname(sandbox_repo)
    shutil.rmtree(tmp_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_api_keys(file_path: str) -> list[str]:
    if not os.path.exists(file_path):
        return ["dummy_key"]
    with open(file_path, "r") as f:
        keys = [line.strip() for line in f if line.strip()]
    return keys if keys else ["dummy_key"]


def read_file_as_string(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def read_jsonl(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_existing_namespaces(file_path: str) -> set[str]:
    if not os.path.exists(file_path):
        return set()
    with open(file_path, "r", encoding="utf-8") as f:
        return {json.loads(line)["metadata"]["namespace"] for line in f if line.strip()}


def append_result(result: dict, file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(metadata: dict) -> str:
    func_name = metadata["namespace"].split(".")[-1]
    rel_path = os.path.relpath(metadata["relative_path"], metadata["repo_path"])
    init = (
        f"Please complete the `{func_name}` function in `{rel_path}` based on its "
        f"docstring and the repository contexts.\n\n"
        f"The `{func_name}` function to complete is as follows:\n\n"
    )
    file_path = os.path.join(REPO_BASE_DIR, metadata["relative_path"])
    code = read_file_as_string(file_path)
    code_lines = code.splitlines()
    start_line_no, end_line_no = metadata["signature_position"]
    signature_lines = code_lines[start_line_no:end_line_no]
    functionality = metadata["requirement"]["Functionality"]
    arguments = metadata["requirement"]["Arguments"]
    requirement = f'"""\n{functionality}\nInput-Output Arguments:\n{arguments}\n"""'
    indent = metadata["indent"]
    requirement_with_indent = textwrap.indent(requirement, " " * indent)
    return (
        init
        + "```python\n"
        + "\n".join(signature_lines)
        + "\n"
        + requirement_with_indent
        + "\n```\n"
    )

def build_prompt_logicoder(metadata, origin_prompt: str):
    func_name = metadata['namespace'].split('.')[-1]
    rel_path = os.path.relpath(metadata['relative_path'], metadata['repo_path'])
    ori_init = f"Please complete the `{func_name}` function based on the relevant snippets and the contexts above the function."
    init = (f"Please complete the `{func_name}` function in `{rel_path}` based on its docstring "
            f"and the repository contexts.\n\nThe `{func_name}` function to complete is as follows:\n\n")
    new_prompt = origin_prompt.replace(ori_init, init)
    final_str = (f"You can navigate the repository for related context, but do NOT run or execute any tests, "
                 f"do NOt ask any question, and do NOT add or modify any code outside the requested function. Finally, "
                 f"please print out the implementation of the target function.")
    return new_prompt + '\n' + final_str


def _extract_python_block(text: str) -> str:
    """Return the last ```python ... ``` block found in text, or empty string."""
    import re
    matches = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    return matches[-1].strip() if matches else ""


# ---------------------------------------------------------------------------
# Agent subclass: treat a no-tool-call response with a python block as a
# clean submission instead of a format error.
# ---------------------------------------------------------------------------

class CodeGenAgent(DefaultAgent):
    """DefaultAgent that accepts a final ```python``` block as a valid submission."""

    def step(self) -> list[dict]:
        try:
            return super().step()
        except FormatError as e:
            # Check if this is the "no tool calls" case (not a parse error).
            fmt_msg = e.messages[0]
            if fmt_msg.get("extra", {}).get("interrupt_type") != "FormatError":
                raise
            # Recover the raw model response stored by LitellmModel.query().
            raw_response = fmt_msg.get("extra", {}).get("response")
            if raw_response is None:
                raise
            # Extract text content from the response object (dict form).
            content = ""
            try:
                content = raw_response["choices"][0]["message"].get("content") or ""
            except (KeyError, IndexError, TypeError):
                pass
            python_block = _extract_python_block(content)
            if not python_block:
                raise
            # Treat the python block as the submission and exit cleanly.
            raise Submitted(
                self.model.format_message(
                    role="exit",
                    content="Submitted",
                    extra={"exit_status": "submitted", "submission": python_block},
                )
            )


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

def make_agent(cwd: str) -> CodeGenAgent:
    """Construct a CodeGenAgent rooted at `cwd` using codegen_config.yaml settings."""
    model_config = dict(_CODE_GEN_CONFIG.get("model", {}))
    env_config = dict(_CODE_GEN_CONFIG.get("environment", {}))

    model = LitellmModel(
        model_name=MODEL,
        cost_tracking="ignore_errors",
        **{k: v for k, v in model_config.items()},
    )
    env = LocalEnvironment(
        cwd=cwd,
        env=env_config.get("env", {}),
    )
    agent_cfg = {
        k: v
        for k, v in _CODE_GEN_CONFIG.get("agent", {}).items()
        if k not in ("mode", "agent_class", "confirm_exit")
    }
    return CodeGenAgent(model, env, **agent_cfg)


def run_mini_query(prompt: str, api_key: str, cwd: str) -> str:
    """Run the agent for one task and return the submitted text."""
    os.environ["DSN_API_KEY"] = api_key
    agent = make_agent(cwd)
    result = agent.run(prompt)
    print(result)
    return result.get("submission", "")


# ---------------------------------------------------------------------------
# Entry processing
# ---------------------------------------------------------------------------

def process_entry(
    entry: dict,
    existing_namespaces: set[str],
    api_key: str,
    retries: int = 3,
):
    """Process one benchmark entry; returns (result_dict, None) or (None, error_str)."""
    if "query_window" in entry:
        entry["metadata"] = entry["query_window"]["metadata"]
        del entry["query_window"]

    namespace = entry["metadata"]["namespace"]

    if namespace in existing_namespaces:
        return None, f"Skipping {namespace}: already completed."

    print(f"[mini-swe-agent] Processing: {namespace}")
    # prompt = build_prompt(entry["metadata"])
    prompt = build_prompt_logicoder(entry["metadata"], entry["instruction_prompt"])

    # print(prompt)
    for attempt in range(1, retries + 1):
        sandbox_repo = setup_sandbox(entry)
        try:
            completion = run_mini_query(prompt, api_key, sandbox_repo)
            result = {
                "prompt": prompt,
                "metadata": entry["metadata"],
                "predictions": [{"text": completion}],
            }
            # print(namespace, ": ", completion)
            return result, None
        except Exception as e:
            wait = 2 ** attempt
            print(
                f"  Attempt {attempt}/{retries} failed for {namespace}: {e}. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)
        finally:
            cleanup_sandbox(sandbox_repo)

    return None, f"Failed to process {namespace} after {retries} retries."


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------

def run_parallel(
    data: list[dict],
    existing_namespaces: set[str],
    api_keys: list[str],
) -> list[dict]:
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                process_entry,
                entry,
                existing_namespaces,
                api_keys[i % len(api_keys)],
            ): entry
            for i, entry in enumerate(data)
        }
        for future in as_completed(futures):
            result, error = future.result()
            if error:
                print(error)
            elif result:
                append_result(result, OUTPUT_FILE)
                results.append(result)
                print(f"  Saved: {result['metadata']['namespace']}")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    api_keys = read_api_keys(API_KEYS_FILE)
    print(f"Loaded {len(api_keys)} API key(s). Model: {MODEL}, Workers: {MAX_WORKERS}")

    data = read_jsonl(PROMPT_FILE)
    existing_namespaces = load_existing_namespaces(OUTPUT_FILE)
    print(f"Total entries: {len(data)}, already done: {len(existing_namespaces)}")

    print(package_dir, _CODE_GEN_CONFIG)

    new_results = run_parallel(data, existing_namespaces, api_keys)
    print(f"Generated {len(new_results)} new completions -> {OUTPUT_FILE}")
