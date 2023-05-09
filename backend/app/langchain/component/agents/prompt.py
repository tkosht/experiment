# flake8: noqa
PREFIX = """Answer the following questions by using tools and thinking step-by-step:"""

FORMAT_INSTRUCTIONS = """# Definition of using the tools
In this conversation, the way you use the tools is by specifying a json blob of the 'Actions' which is defined as bellow, and also even if 'execution' in Japanese and so on.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).
And also define the successful of 'using the tools', 'execution' is the meaning no error or successful of 'Observation' which is defined as below.

# Definition of $JSON_BLOB of a 'Action'

The only values that MUST be in the "action" field are: {tool_names}

The $JSON_BLOB should only contain a SINGLE action, do NOT return any comments. Here is an example of a valid $JSON_BLOB:

```
{{{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}}}
```

after you responding 'Action', you wait for the content of 'Observation' (the result of action(execution of tools))


# Definition of the format

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about what to do by using above tools step-by-step
Action:
```
$JSON_BLOB
```
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final direct answer to the original input question by using the tools above
"""

SUFFIX = ("""
Note that, Especially, NEVER answer like that `I cannot directly download webpages or execute shell commands, python codes`
instead, NONEEDS any alternative proposal and to excuse not to answer directly, but you HAVE TO use the tools above to answer HUMAN directly.

Remember! MUST always use the exact characters `Final Answer` when responding.
the phrases, Thought/Action/Observation/Final Answer, are MUST NOT be translated in Japanese.
if you found error, MUST fix error with step-by-step 'Thought:' in order not to repeat getting same errors
Let's Begin!
""")

# ====================================================================================================
# # below is original definition
#
# PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
# FORMAT_INSTRUCTIONS = """The way you use the tools is by specifying a json blob.
# Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).
#
# The only values that should be in the "action" field are: {tool_names}
#
# The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:
#
# ```
# {{{{
#   "action": $TOOL_NAME,
#   "action_input": $INPUT
# }}}}
# ```
#
# ALWAYS use the following format:
#
# Question: the input question you must answer
# Thought: you should always think about what to do
# Action:
# ```
# $JSON_BLOB
# ```
# Observation: the result of the action
# ... (this Thought/Action/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question"""
# SUFFIX = """Begin! Reminder to always use the exact characters `Final Answer` when responding."""

# NOTE: cf. https://github.com/hwchase17/langchain/blob/master/langchain/agents/chat/prompt.py
# Copyright (c) Harrison Chase
# Copyright (c) 2023 Takehito Oshita
