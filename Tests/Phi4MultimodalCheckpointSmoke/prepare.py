"""Native checkpoint tokenizer contracts, image and audio for the opt-in smoke."""
import json,re,sys,wave
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from transformers import PreTrainedTokenizerFast
checkpoint,output=map(Path,sys.argv[1:]);output.mkdir(parents=True,exist_ok=True)
tokenizer=PreTrainedTokenizerFast.from_pretrained(checkpoint,local_files_only=True)
image=Image.new('RGB',(64,64),'white');ImageDraw.Draw(image).text((20,20),'7',fill='black');image.save(output/'image.png')
samples=(np.sin(np.arange(16000)*2*np.pi*440/16000)*0.1+np.random.default_rng(719).normal(0,.01,16000))
with wave.open(str(output/'audio.wav'),'wb') as file:
    file.setnchannels(1);file.setsampwidth(2);file.setframerate(16000);file.writeframes((samples*32767).astype('<i2').tobytes())
prompts=['What number is shown?','Describe the sound.','What is 3+4? Answer with the number only.']
fixture={'image':str((output/'image.png').resolve()),'prompts':prompts,'encodings':{},'templates':{},'specialTokens':{}}
gemma='gemma' in str(Path(__file__).parent).lower()
for index,(kind,prompt) in enumerate(zip(['image','audio','text'],prompts)):
    if gemma:
        content=([{'type':kind}] if kind!='text' else [])+[{'type':'text','text':prompt}]
    else:
        content=(('<|image_1|>\n' if kind=='image' else '<|audio_1|>\n') if kind!='text' else '')+prompt
    rendered=tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
    ids=tokenizer.encode(rendered,add_special_tokens=False)
    decoded=tokenizer.decode(ids,skip_special_tokens=False)
    fixture['templates'][kind]={'text':decoded,'tokens':ids}
    normalized=re.sub(r'<\|image_\d+\|>','<image>',decoded)
    normalized=re.sub(r'<\|audio_\d+\|>','<|endoftext11|>',normalized)
    for piece in re.split(r'<image>|<\|endoftext11\|>',normalized):
        fixture['encodings'][piece]=tokenizer.encode(piece,add_special_tokens=False)
fixture['encodings']['\n\n']=tokenizer.encode('\n\n',add_special_tokens=False)
for token_id,token in tokenizer.added_tokens_decoder.items(): fixture['specialTokens'][token.content]=token_id
(output/'input.json').write_text(json.dumps(fixture))
