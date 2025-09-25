import sentencepiece as spm, os
p = r'D:\A-Coding Projects\Wan2.2\Wan2.2-Animate-14B\process_checkpoint\FLUX.1-Kontext-dev\tokenizer_2\spiece.model'
print('spiece exists:', os.path.isfile(p))
sp = spm.SentencePieceProcessor()
ok = sp.Load(p)
print('Load OK:', ok, 'pieces:', sp.GetPieceSize())
