"""Generate native chat-template/tokenizer contracts and a 64px image for MiniCPM smoke."""
import json,re,sys
from pathlib import Path
from PIL import Image,ImageDraw
from transformers import AutoTokenizer
checkpoint,output=map(Path,sys.argv[1:]);output.mkdir(parents=True,exist_ok=True)
tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False)
image=Image.new('RGB',(64,64),'white');ImageDraw.Draw(image).text((20,20),'7',fill='black');image.save(output/'image.png')
prompts=['What number is shown? Answer with the number only.','What is 3+4? Answer with a single number.']
fixture={'image':str((output/'image.png').resolve()),'prompts':prompts,'encodings':{},'templates':{},'specialTokens':{}}
for index,prompt in enumerate(prompts):
    content=('<image>' if index==0 else '')+prompt
    text=tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=tokenizer.encode(text,add_special_tokens=False)
    fixture['templates'][content]={'text':tokenizer.decode(ids,skip_special_tokens=False),'tokens':ids}
    text=fixture['templates'][content]['text'].replace('<think>\n\n</think>\n\n','')
    for piece in re.split(r'<image>\./</image>|<image>',text):
        fixture['encodings'][piece]=tokenizer.encode(piece,add_special_tokens=False)
for text in ['<image_id>0</image_id>','\n']:
    fixture['encodings'][text]=tokenizer.encode(text,add_special_tokens=False)
for text in ['<image>','</image>','<slice>','</slice>']:
    ids=tokenizer.encode(text,add_special_tokens=False);assert len(ids)==1,(text,ids);fixture['specialTokens'][text]=ids[0]
(output/'input.json').write_text(json.dumps(fixture))
print(output/'input.json')
