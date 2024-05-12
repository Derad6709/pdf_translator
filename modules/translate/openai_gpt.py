from .base import TranslateBase 
from openai import OpenAI
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100-12B-avg-5-ckpt")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100-12B-avg-5-ckpt")


def system_prompt(from_lang, to_lang):
    p  = "You are an %s-to-%s translator. " % (from_lang, to_lang)
    p += "Keep all special characters and HTML tags as in the source text. Return only %s translation." % to_lang
    return p

langs = ['af', 'am', 'ar', 'ast', 'az', 'ba', 'be', 'bg', 'bn', 'br', 'bs', 'ca', 'ceb', 'cs', 'cy', 'da', 'de', 'el', 'en', 'es', 'et', 'fa', 'ff', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'he', 'hi', 'hr', 'ht', 'hu', 'hy', 'id', 'ig', 'ilo', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'lb', 'lg', 'ln', 'lo', 'lt', 'lv', 'mg', 'mk', 'ml', 'mn', 'mr', 'ms', 'my', 'ne', 'nl', 'no', 'ns', 'oc', 'or', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sd', 'si', 'sk', 'sl', 'so', 'sq', 'sr', 'ss', 'su', 'sv', 'sw', 'ta', 'th', 'tl', 'tn', 'tr', 'uk', 'ur', 'uz', 'vi', 'wo', 'xh', 'yi', 'yo', 'zh', 'zu']
class TranslateOpenAIGPT(TranslateBase):
    def init(self, cfg: dict):
        self.client = OpenAI(api_key=cfg['openai_api_key'])

    def get_languages(self):
        return langs

    def translate(self, text: str, from_lang='en', to_lang='si') -> str:
        tokenizer.src_lang = text
        encoded_hi = tokenizer(hi_text, return_tensors="pt")
        generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id(to_lang))
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return translated_text
