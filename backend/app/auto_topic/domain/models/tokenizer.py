from typing import TypeAlias

# from app.auto_topic.component.models.tokenizer import JpTokenizerJanome
from app.auto_topic.component.models.tokenizer import JpTokenizerMeCab

TokenizerWord: TypeAlias = JpTokenizerMeCab
