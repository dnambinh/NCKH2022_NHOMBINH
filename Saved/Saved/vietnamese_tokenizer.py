from __future__ import annotations
from typing import Any, Dict, List, Text, Optional

import rasa.shared.utils.io
import rasa.utils.io

from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.constants import DOCS_URL_COMPONENTS
from rasa.shared.nlu.training_data.message import Message

from underthesea import word_tokenize
import pycountry
import regex

@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER, is_trainable = False
)
class VietnameseTokenizer(Tokenizer):
    @staticmethod
    def not_supported_languages() -> Optional[List[Text]]:
        """The languages that are not supported."""
        all_langs = [country.alpha_2.lower() for country in pycountry.countries]
        all_langs.remove('vi')
        return all_langs

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the component's default config."""
        return {
            # This *must* be added due to the parent class.
            "intent_tokenization_flag": False,
            # This *must* be added due to the parent class.
            "intent_split_symbol": "_",
            # Regular expression to detect tokens
            "token_pattern": None,
            # This is the spaCy language setting.
            "case_sensitive": True,
        }

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> VietnameseTokenizer:
        """Creates a new component (see parent class for full docstring)."""
        # Path to the dictionaries on the local filesystem.
        return cls(config)

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ['underthesea']

    def __init__(self, config: Dict[Text, Any]) -> None:
        """Initialize the tokenizer."""
        super().__init__(config)
        self.emoji_pattern = rasa.utils.io.get_emoji_regex()
        self.case_sensitive = config['case_sensitive']

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        text = message.get(attribute)
        text = text.replace('\n', ' ')
        text = text.replace('òa', 'oà').replace('óa', 'oá').replace('ỏa', 'oả').replace('õa', 'oã').replace('ọa', 'oạ').replace('òe', 'oè').replace('óe', 'oé').replace('ỏe', 'oẻ').replace('õe', 'oẽ').replace('ọe', 'oẹ').replace('ùy', 'uỳ').replace('úy', 'uý').replace('ủy', 'uỷ').replace('ũy', 'uỹ').replace('ụy', 'uỵ')
        text = regex.sub(
            # there is a space or an end of a string after it
            r"[^\w#@&]+(?=\s|$)|"
            # there is a space or beginning of a string before it
            # not followed by a number
            r"(\s|^)[^\w#@&]+(?=[^0-9\s])|"
            # not in between numbers and not . or @ or & or - or #
            # e.g. 10'000.00 or blabla@gmail.com
            # and not url characters
            r"(?<=[^0-9\s])[^\w._~:/?#\[\]()@!$&*+,;=-]+(?=[^0-9\s])",
            ' ',
            text,
        )
        text = text.replace('  ', ' ').strip()

        words = word_tokenize(text)
        if not words:
            words = [text]
        
        tokens = self._convert_words_to_tokens(words, text)
        return self._apply_token_pattern(tokens)