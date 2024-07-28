from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import re
from tqdm import tqdm

def replace_consecutve_whitespaces(input_string:str):
    return re.sub(r'\s+', ' ', input_string)

def process_string(input_string:str):
    # Find all hashtags
    hashtags = re.findall(r'<hashtag>(.*?)</hashtag>', input_string)
    
    # Split the string by hashtags
    parts = re.split(r'<hashtag>.*?</hashtag>', input_string)
    
    # Remove empty strings from parts
    parts = [part.strip() for part in parts if part.strip()]
    
    # Format the output list
    hashtags_written_out = input_string.replace('<hashtag> ', '').replace('</hashtag>', '')
    no_hashtags = ' '.join(parts) if '<hashtag>' in input_string else ''
    output = [replace_consecutve_whitespaces(hashtags_written_out)] + [replace_consecutve_whitespaces(no_hashtags)] + [hashtags]
    
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
    pre_process_1 = []
    for i in tqdm(range(len(tweets))):
        pre_process_1.append(" ".join(text_processor.pre_process_doc(tweets[i].replace('\n', ''))))
    pre_process_2 = []
    for i in tqdm(range(len(pre_process_1))):
        pre_process_2.append(process_string(pre_process_1[i]))
    
    return pre_process_2
    
def main():
    print(preprocess(["I love #pizza", "I love #pasta", "I love #food"]))
    print(emoticons)

if __name__ == '__main__':
    main()