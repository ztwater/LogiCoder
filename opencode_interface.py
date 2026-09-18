import asyncio
import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from opencode_agent_sdk import SDKClient, AgentOptions, AssistantMessage, TextBlock

# ============================== Configuration ==============================
MODEL = "deepseek-v4-pro"
# MODEL = "gpt-4o"

# Multiple API keys for parallel execution (loaded from config)
API_KEYS_FILE = "/media/hd/RepoLevelGen/DevEval/config/api_keys_deepseek.txt"

PROMPT_FILE = "/media/hd/RepoLevelGen/DevEval/prompts/logicoder/logicoder-agent.jsonl"
OUTPUT_FILE = f"./results/completion_opencode_logicoder_{MODEL}_T0_N1.jsonl"

REPO_BASE_DIR = "/media/hd/RepoLevelGen/DevEval/Source_Code"

MAX_WORKERS = 4
# Base port for the worker server pool; workers use BASE_PORT + worker_index
BASE_PORT = 34000
# Seconds to wait for an opencode serve instance to become ready
SERVER_READY_TIMEOUT = 30
# ===========================================================================

SYSTEM_PROMPT = (
    "You are an expert Python developer. Your task is to implement the specified function "
    "in the given repository. You can navigate the repository for related context, but "
    "do NOT run or execute any tests, and do NOT add any code outside the requested function. "
    "Return only the implementation of the target function."
)


# ---------------------------------------------------------------------------
# Per-worker opencode serve pool
# ---------------------------------------------------------------------------

class WorkerServer:
    """A single opencode serve process pinned to a port and a working directory."""

    def __init__(self, worker_id: int, port: int):
        self.worker_id = worker_id
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self._proc: subprocess.Popen | None = None
        self._current_cwd: str | None = None

    def _is_port_free(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", self.port)) != 0

    def _wait_ready(self, timeout: float = SERVER_READY_TIMEOUT) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                r = httpx.get(f"{self.url}/api/session", timeout=2.0)
                if r.status_code < 500:
                    return
            except Exception:
                pass
            time.sleep(0.5)
        raise RuntimeError(
            f"Worker {self.worker_id}: opencode serve on port {self.port} "
            f"did not become ready within {timeout}s"
        )

    def start(self, cwd: str) -> None:
        """Start (or restart) the server with the given working directory."""
        self.stop()
        if not self._is_port_free():
            raise RuntimeError(
                f"Worker {self.worker_id}: port {self.port} is already in use"
            )
        print(f"  [worker {self.worker_id}] Starting opencode serve "
              f"on port {self.port} in {cwd}")
        self._proc = subprocess.Popen(
            ["opencode", "serve", "--port", str(self.port)],
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._current_cwd = cwd
        self._wait_ready()

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None
            self._current_cwd = None

    def ensure_cwd(self, cwd: str) -> None:
        """Restart the server only when the working directory changes."""
        if self._proc is None or self._current_cwd != cwd:
            self.start(cwd)


class WorkerPool:
    """Pool of WorkerServer instances handed out via a thread-safe queue."""

    def __init__(self, n_workers: int, base_port: int):
        self._servers = [
            WorkerServer(i, base_port + i) for i in range(n_workers)
        ]
        self._queue: queue.Queue[WorkerServer] = queue.Queue()
        for s in self._servers:
            self._queue.put(s)

    def acquire(self) -> WorkerServer:
        return self._queue.get()

    def release(self, server: WorkerServer) -> None:
        self._queue.put(server)

    def shutdown(self) -> None:
        for s in self._servers:
            s.stop()


# ---------------------------------------------------------------------------
# Sandbox helpers
# ---------------------------------------------------------------------------

def setup_sandbox(entry: dict) -> str:
    """Copy the target repo to a temp dir and mask the function body with `pass`.

    Returns the path to the sandbox repo root (equivalent to the original cwd).
    The caller is responsible for deleting it via cleanup_sandbox().
    """
    metadata = entry["metadata"]
    repo_path = metadata.get("repo_path", "")
    relative_path = metadata["relative_path"]   # e.g. "Internet/boto/boto/datapipeline/__init__.py"
    # body_position is [def_line, last_body_line], both 0-based inclusive.
    # body_position[0] points to the `def` line itself; the actual body starts
    # on the next line.
    body_start, body_end = metadata["body_position"]

    src_repo = os.path.join(REPO_BASE_DIR, repo_path)

    # Create a temp directory and copy the entire repo into it.
    # The sandbox root mirrors src_repo exactly.
    tmp_root = tempfile.mkdtemp(prefix="opencode_sandbox_")
    sandbox_repo = os.path.join(tmp_root, os.path.basename(src_repo))
    shutil.copytree(src_repo, sandbox_repo, symlinks=True)

    # Mask the function body.
    # relative_path is relative to REPO_BASE_DIR, so strip the repo_path prefix
    # to get the path within the repo.
    rel_in_repo = os.path.relpath(
        os.path.join(REPO_BASE_DIR, relative_path), src_repo
    )
    target_file = os.path.join(sandbox_repo, rel_in_repo)

    with open(target_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Determine indentation from the first actual body line; fall back to metadata.
    indent = metadata.get("indent", 4)
    if body_start < len(lines):
        first_body_line = lines[body_start]
        stripped = first_body_line.lstrip()
        if stripped:
            indent = len(first_body_line) - len(stripped)

    # Keep the `def` line, replace all body lines with a single `pass`.
    masked = (
        lines[:body_start]
        + [" " * indent + "pass\n"]
        + lines[body_end:]          # body_end is inclusive, so skip through it
    )

    with open(target_file, "w", encoding="utf-8") as f:
        f.writelines(masked)

    return sandbox_repo


def cleanup_sandbox(sandbox_repo: str) -> None:
    """Delete the temp directory created by setup_sandbox."""
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
    with open(file_path, 'r', encoding="utf-8") as f:
        res = f.read()
    return res

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
# Query
# ---------------------------------------------------------------------------

async def run_opencode_query(prompt: str, server_url: str) -> str:
    """Run a single query against a per-worker opencode serve instance."""
    options = AgentOptions(
        model=MODEL,
        server_url=server_url,
        system_prompt=SYSTEM_PROMPT,
    )

    client = SDKClient(options=options)
    collected: list[str] = []

    try:
        await client.connect()
        await client.query(prompt)

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        collected.append(block.text)
    finally:
        await client.disconnect()

    return "".join(collected)


# ---------------------------------------------------------------------------
# Entry processing
# ---------------------------------------------------------------------------

def process_entry(
    entry: dict,
    existing_namespaces: set[str],
    api_key: str,
    pool: WorkerPool,
    retries: int = 3,
):
    """Process a single benchmark entry using a sandboxed per-job repo copy."""
    if "query_window" in entry:
        entry["metadata"] = entry["query_window"]["metadata"]
        del entry["query_window"]
    namespace = entry["metadata"]["namespace"]
    # prompt = build_prompt(entry["metadata"])
    prompt = build_prompt_logicoder(entry["metadata"], entry["instruction_prompt"])
    # print(prompt)
    if namespace in existing_namespaces:
        return None, f"Skipping {namespace}: already completed."

    print(f"[opencode] Processing: {namespace}")

    # Each job gets its own copy of the repo with the target function masked.
    # The original REPO_BASE_DIR is never modified.
    sandbox_repo = setup_sandbox(entry)
    server = pool.acquire()
    try:
        server.ensure_cwd(sandbox_repo)

        for attempt in range(1, retries + 1):
            try:
                completion = asyncio.run(run_opencode_query(prompt, server.url))
                # The model echoes the input prompt at the start of its response.
                # Strip it so predictions contain only the model's own output.
                if completion.startswith(prompt):
                    completion = completion[len(prompt):]
                result = {
                    "prompt": prompt,
                    "metadata": entry["metadata"],
                    "predictions": [{"text": completion}],
                }
                return result, None
            except Exception as e:
                wait = 2 ** attempt
                print(f"  Attempt {attempt}/{retries} failed for {namespace}: {e}. "
                      f"Retrying in {wait}s...")
                time.sleep(wait)
                # Rebuild the sandbox and restart the server on failure.
                cleanup_sandbox(sandbox_repo)
                sandbox_repo = setup_sandbox(entry)
                try:
                    server.start(sandbox_repo)
                except Exception as restart_err:
                    print(f"  Server restart failed: {restart_err}")
    finally:
        pool.release(server)
        cleanup_sandbox(sandbox_repo)

    return None, f"Failed to process {namespace} after {retries} retries."


def run_parallel(data: list[dict], existing_namespaces: set[str], api_keys: list[str]) -> list[dict]:
    pool = WorkerPool(n_workers=MAX_WORKERS, base_port=BASE_PORT)
    results = []
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    process_entry,
                    entry,
                    existing_namespaces,
                    api_keys[i % len(api_keys)],
                    pool,
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
    finally:
        pool.shutdown()

    return results

def build_prompt(metadata):
    func_name = metadata['namespace'].split('.')[-1]
    rel_path = os.path.relpath(metadata['relative_path'], metadata['repo_path'])
    init = (f"Please complete the `{func_name}` function in `{rel_path}` based on its docstring "
            f"and the repository contexts.\n\nThe `{func_name}` function to complete is as follows:\n\n")
    file_path = os.path.join(REPO_BASE_DIR, metadata['relative_path'])
    code = read_file_as_string(file_path)
    code_lines = code.splitlines()
    start_line_no, end_line_no = metadata['signature_position']
    signature_lines = code_lines[start_line_no:end_line_no]
    functionality = metadata['requirement']['Functionality']
    arguments = metadata['requirement']['Arguments']
    requirement = f"\"\"\"\n{functionality}\nInput-Output Arguments:\n{arguments}\n\"\"\""
    indent = metadata['indent']
    requirement_with_indent = textwrap.indent(requirement, ' ' * indent)
    final_str = (f"You can navigate the repository for related context, but do NOT run or execute any tests, "
                 f"do NOt ask any question, and do NOT add or modify any code outside the requested function. Finally, "
                 f"please print out the implementation of the target function.")

    return init + '```python\n' + '\n'.join(signature_lines) + '\n' + requirement_with_indent + '\n```\n\n' + final_str

def build_prompt_logicoder(metadata, origin_prompt: str):
    func_name = metadata['namespace'].split('.')[-1]
    if metadata['repo_path'].startswith('/media'):
        metadata['repo_path'] = metadata['repo_path'].replace('/media/hd/RepoLevelGen/DevEval/Source_Code/', '')
    rel_path = os.path.relpath(metadata['relative_path'], metadata['repo_path'])
    ori_init = f"Please complete the `{func_name}` function based on the relevant snippets and the contexts above the function."
    init = (f"Please complete the `{func_name}` function in `{rel_path}` based on its docstring "
            f"and the repository contexts.\n\nThe `{func_name}` function to complete is as follows:\n\n")
    new_prompt = origin_prompt.replace(ori_init, init)
    # file_path = os.path.join(REPO_BASE_DIR, metadata['relative_path'])
    # code = read_file_as_string(file_path)
    # code_lines = code.splitlines()
    # start_line_no, end_line_no = metadata['signature_position']
    # signature_lines = code_lines[start_line_no:end_line_no]
    # functionality = metadata['requirement']['Functionality']
    # arguments = metadata['requirement']['Arguments']
    # requirement = f"\"\"\"\n{functionality}\nInput-Output Arguments:\n{arguments}\n\"\"\""
    # indent = metadata['indent']
    # requirement_with_indent = textwrap.indent(requirement, ' ' * indent)
    final_str = (f"You can navigate the repository for related context, but do NOT run or execute any tests, "
                 f"do NOt ask any question, and do NOT add or modify any code outside the requested function. Finally, "
                 f"please print out the implementation of the target function.")

    return new_prompt + '\n' + final_str

if __name__ == "__main__":
    api_keys = read_api_keys(API_KEYS_FILE)
    print(f"Loaded {len(api_keys)} API key(s). Model: {MODEL}, Workers: {MAX_WORKERS}")

    data = read_jsonl(PROMPT_FILE)
    existing_namespaces = load_existing_namespaces(OUTPUT_FILE)
    print(f"Total entries: {len(data)}, already done: {len(existing_namespaces)}")

    new_results = run_parallel(data, existing_namespaces, api_keys)
    print(f"Generated {len(new_results)} new completions -> {OUTPUT_FILE}")
