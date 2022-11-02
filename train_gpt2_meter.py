import os
#os.environ["CURL_CA_BUNDLE"]=""
import trlx
from trlx.model.accelerate_ppo_model import AcceleratePPOModel
from transformers import AutoTokenizer
from trlx.utils.loading import get_model, get_orchestrator
from trlx.data.configs import TRLConfig
from count_syllables import count_syllables_in_haiku
import math
import json


def compute_meter_score(sample, max_violations=4., parseable=0.2, min_for_correct_lines=0.2, return_explanation=False):
    score = 0.
    explanation = ''  # could also be more detailed
    if '|' in sample:
        explanation = 'triggered phoneme mode'
    elif sample != '<syntax_error>':
        # Is at least parseable
        score += parseable
        syllables = count_syllables_in_haiku(sample, fast=False)
        #print(syllables, sample)
        is_three_lines = len(syllables) == 3
        if is_three_lines:
            # Got three lines
            score += min_for_correct_lines
            violations = abs(syllables[0]-5) + abs(syllables[1]-7) + abs(syllables[2]-5)
            if syllables[0]>5:
                explanation += 'line 0 is longer than 5,'
            elif syllables[0]<5:
                explanation += 'line 0 is shorter than 5,'
            if syllables[1]>7:
                explanation += 'line 1 is longer than 7,'
            elif syllables[1]<7:
                explanation += 'line 1 is shorter than 7,'
            if syllables[2]>5:
                explanation += 'line 2 is longer than 5,'
            elif syllables[2]<5:
                explanation += 'line 2 is shorter than 5,'
            # Total score
            score += (1-min_for_correct_lines-parseable)*(1. - min(1., violations/max_violations))
            explanation = 'perfect'
        else:
            explanation = 'number of lines is not 3'
    else:
        explanation = 'syntax error'
    if return_explanation:
        return score, explanation
    else:
        return score
        
def process_output(sample):
    # Stop at the first closing parenthesis
    parts = sample.split(')')
    if len(parts)>=1:
        tokens = parts[0].split(' = ')
        if len(tokens) == 2:
            return tokens[1]
    # in any other case return the empty string
    return '<syntax_error>'

def compute_score(sample):
    processed = process_output(sample)
    score, explanation = compute_meter_score(processed, return_explanation=True)
    return score


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
topics = []
with open('data/reddit_dataset.json') as fp:
    reddit_dataset = json.load(fp)
    for x in reddit_dataset['data']:
        topics.append(next(iter(x)))
prompts = [f'({topic} = ' for topic in topics]
#prompts = [f'<|endoftext|>({topic} = ' for topic in topics]
config = TRLConfig.load_yaml("configs/ppo_config_gpt2_haiku.yml")
model = trlx.train("fabianmmueller/deep-haiku-gpt-2", 
    reward_fn=lambda samples: [compute_score(sample) for sample in samples],
    prompts=prompts,
    eval_prompts=prompts[:10],  # every 50
    config=config)

'''
# Save model, optimizer, scheduler ... 
model.save('model_meter_deep_haiku.pt')


# Reload saved model
config = TRLConfig.load_yaml("configs/ppo_config.yml")
config.model.model_path = 'gpt2'
model = get_model(config.model.model_type)(config)
model.accelerator.load_state('model.pt')

tokenizer = model.tokenizer
print(tokenizer.decode(model.generate(**tokenizer('Q: Who rules the world? A:', return_tensors='pt'), do_sample=True)[0]))
'''