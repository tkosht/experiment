# flake8: noqa
PREFIX = """Answer the following questions as best you can.
but you NEVER answer like that `I'm sorry, but I cannot directly download webpages or execute shell commands like grep`
You have to access to the following tools:"""

FORMAT_INSTRUCTIONS = """The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

The only values that MUST be in the "action" field are: {tool_names}

The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

```
{{{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}}}
```

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about what to do step-by-step
Action:
```
$JSON_BLOB
```
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final direct answer to the original input question"""

SUFFIX = ("Begin! but remember! MUST always use the exact characters `Final Answer` when responding. "
         "thease phrases, Thought/Action/Observation/Final Answer, are MUST NOT be translated in Japanese. "
         "if you found error, MUST fix error to success with step-by-step your 'Thought:'"
         )
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
