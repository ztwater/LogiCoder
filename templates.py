COT_PROMPT_TEMPLATE = {
    "system_prompt": """\
You are a powerful Python programming problem solver. Your task is to generate a plan to divide a Python \
function-level programming task into steps based on its signature and code context.""",

    "instruction_prompt": """\
Please generate a plan for the given function `{function_name}` to divide a function-level task into steps based on \
the following examples and the file context of the function.

# Here are some examples:
{demonstrations}
{context_above}
# The `{function_name}` function to complete is {function_type_desc}as follows:
{function_signature}

Note that:
- Your plan should be a list of strings where each represents a single step.
- Your plan should describe the solution to the problem at the high level, where each step should describe what to do \
and for what purpose.
- Your plan should be concise and contain as few steps as possible.
- Each step in your plan should be written as ONE verb-object phrase.
- Each step in your plan should NOT contain titles as well as any markdown, latex or html syntax.

# Your Answer:
"""
}


COT_AND_CODE_PROMPT_TEMPLATE = {
    "system_prompt": """\
You are a powerful Python programming problem solver. Your task is to first generate a plan to divide a Python \
function-level programming task into steps based on its signature and code context and then complete the code based on \
your plan and the contexts above the function.""",

    "instruction_prompt": """\
Please generate a step-wise plan for the given function `{function_name}` to divide a function-level task into steps \
and then complete it based on the following examples and the file context of the function.

# Here are some examples:
{demonstrations}
{context_above}
# The `{function_name}` function to complete is {function_type_desc}as follows:
{function_signature}

Note that:
- Your plan should be written as a numbered-list in the docstring where each item represents a single step.
- Your plan should describe the solution to the problem at the high level, where each step should describe what to do \
and for what purpose.
- Your plan should be concise and contain as few steps as possible.
- Each step in your plan should be written as ONE verb-object phrase.
- Each step in your plan should NOT contain titles as well as any markdown, latex or html syntax.

# Your Answer:
"""
}


SELF_PLANNING_PROMPT_TEMPLATE = {
    "system_prompt": """\
You are a powerful Python programmer. Your responsibility is to complete a function-level code \
based on the plan and the contexts above the function.""",

    "instruction_prompt": """\
Please complete the `{function_name}` function based on the plan and the file context of the function.
{context_above}
# The `{function_name}` function to complete is {function_type_desc}as follows:
{function_signature}

"""
}


RAG_PROMPT_TEMPLATE = {
    'system_prompt': """You are a powerful Python programmer. Your responsibility is to complete a function-level code \
based on the contexts above the function.""",

    'instruction_prompt': """\
Please complete the `{function_name}` function based on the relevant snippets and the contexts above the function.

{retrieved_blocks}
{context_above}
The `{function_name}` function to complete is {function_type_desc}as follows:
{function_signature}

"""
}


DIRECT_PROMPT_TEMPLATE = {
    'system_prompt': """You are a powerful Python programmer. Your responsibility is to complete a function-level code \
based on some relevant snippets collected from local repository context and the contexts above the function.""",

    'instruction_prompt': """\
Please complete the `{function_name}` function (`{file_path}`) based on the contexts above the function.
{context_above}
# The `{function_name}` function to complete is {function_type_desc}as follows:
{function_signature}

"""
}


LONG_CONTEXT_PROMPT_TEMPLATE = {
    'system_prompt': """You are a powerful Python programmer. Your responsibility is to complete a function-level code \
based on code collected from local repository context and the contexts above the function.""",

    'instruction_prompt': """\
Please complete the `{function_name}` function (`{file_path}`) based on the repository context.
{repo_files}
# The context above the target function (`{file_path}`) is as follows:
{context_above}
# The `{function_name}` function to complete is {function_type_desc}as follows:
{function_signature}

"""
}
