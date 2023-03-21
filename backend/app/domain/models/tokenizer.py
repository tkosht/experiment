from typing import TypeAlias

# from app.component.models.tokenizer import JpTokenizerJanome
from app.component.models.tokenizer import JpTokenizerMeCab

TokenizerWord: TypeAlias = JpTokenizerMeCab
