import os
#os.environ["CURL_CA_BUNDLE"]=""
import trlx
from trlx.model.accelerate_ppo_model import AcceleratePPOModel
from transformers import AutoTokenizer
from trlx.utils.loading import get_model, get_orchestrator
from trlx.data.configs import TRLConfig
from count_syllables import count_syllables_in_haiku
import math


def compute_meter_score(sample, max_violations=4., min_for_correct_lines=0.2):
    syllables = count_syllables_in_haiku(sample)
    print(syllables, sample)
    is_three_lines = len(syllables) == 3
    if is_three_lines:
        violations = abs(syllables[0]-5) + abs(syllables[1]-7) + abs(syllables[2]-5)
        # Total score
        return min_for_correct_lines + (1-min_for_correct_lines)*(1. - min(1., violations/max_violations))
    else:
        return 0.
        

# Try out the haiku meter counter
haikus = ['Deep autumn; /	The apple colder/	In the tree.',
"It's crazy to see / How small my body can get? / And still feel complete",
"No body gonna / Help you on Twitter until / You can get your shit",
"First haiku... unintentionally reversed (7 / 5 / 7)",
"I predicted this / Today's result, but it will / Still be reversed",
"Too / many lines / there is / here"
]
for haiku in haikus:
    print(haiku, compute_meter_score(haiku))

# The model needs to be conditioned on topics,
# it's not good unconditionally and will produce gibberish (corresponding to the training)
model = trlx.train("fabianmmueller/deep-haiku-gpt-2", reward_fn=lambda samples: [compute_meter_score(sample) for sample in samples])


# Save model, optimizer, scheduler ... 
model.save('model.pt')

# Reload saved model
config = TRLConfig.load_yaml("configs/ppo_config.yml")
config.model.model_path = 'gpt2'
model = get_model(config.model.model_type)(config)
model.accelerator.load_state('model.pt')

tokenizer = model.tokenizer
print(tokenizer.decode(model.generate(**tokenizer('Q: Who rules the world? A:', return_tensors='pt'), do_sample=True)[0]))