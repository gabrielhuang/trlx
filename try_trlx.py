import os
#os.environ["CURL_CA_BUNDLE"]=""
import trlx
from trlx.model.accelerate_ppo_model import AcceleratePPOModel
from transformers import AutoTokenizer
from trlx.utils.loading import get_model, get_orchestrator
from trlx.data.configs import TRLConfig


model = trlx.train('gpt2', reward_fn=lambda samples: [sample.count('cats') for sample in samples])

# Save model, optimizer, scheduler ... 
model.save('model.pt')

# Reload saved model
config = TRLConfig.load_yaml("configs/ppo_config.yml")
config.model.model_path = 'gpt2'
model = get_model(config.model.model_type)(config)
model.accelerator.load_state('model.pt')

tokenizer = model.tokenizer
print(tokenizer.decode(model.generate(**tokenizer('Q: Who rules the world? A:', return_tensors='pt'), do_sample=True)[0]))