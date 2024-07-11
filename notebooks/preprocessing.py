from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import re

def process_string(input_string:str):
    # Find all hashtags
    hashtags = re.findall(r'<hashtag>(.*?)</hashtag>', input_string)
    
    # Split the string by hashtags
    parts = re.split(r'<hashtag>.*?</hashtag>', input_string)
    
    # Remove empty strings from parts
    parts = [part.strip() for part in parts if part.strip()]
    
    # Format the output list
    hashtags_written_out = input_string.replace('<hashtag> ', '').replace('</hashtag> ', '')
    no_hashtags = ''.join(parts) if '<hashtag>' in input_string else ''
    output = [hashtags_written_out] + [no_hashtags] + [hashtags]
    
    return output

def preprocess(tweets:list, normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'date', 'number']):
    text_processor = TextPreProcessor(
    fix_html=True,
    segmenter="twitter", 
    corrector="twitter", 
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[emoticons],
    
    unpack_hashtags=True,
    unpack_contractions=True,
    spell_correct_elong=True,
    normalize=normalize,
    annotate={'hashtag'},
    )
    pre_process_1 = [" ".join(text_processor.pre_process_doc(tweet)) for tweet in tweets]
    pre_process_2 = [process_string(tweet) for tweet in pre_process_1]
    
    return pre_process_2
    
