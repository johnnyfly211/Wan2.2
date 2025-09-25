import os, traceback
from transformers import T5Tokenizer
p = r'D:\A-Coding Projects\Wan2.2\Wan2.2-Animate-14B\process_checkpoint\FLUX.1-Kontext-dev\tokenizer_2\spiece.model'
print('exists:', os.path.isfile(p), 'size:', os.path.getsize(p) if os.path.isfile(p) else 'NA')
try:
    tok = T5Tokenizer(vocab_file=p, extra_ids=0)
    print('OK: vocab_size', tok.vocab_size)
except Exception as e:
    print('ERROR:', repr(e))
    traceback.print_exc()
