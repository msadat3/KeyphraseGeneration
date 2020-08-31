import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('D:\\Research\\KeyphraseGeneration\\transformers_master\\src\\')
print(sys.path)
from transformers_a.modeling_bart import BartForConditionalGeneration
#from . import transformer_master

model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
model.print_test()