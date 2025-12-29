from aerial_gym.mav_baselines.torch.recurrent_ppo.recurrent.policies import (
    RecurrentActorCriticCnnPolicy,
    RecurrentActorCriticPolicy,
    RecurrentMultiInputActorCriticPolicy,
)
from aerial_gym.mav_baselines.torch.recurrent_ppo.recurrent.policies import register_policy

from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy, 
    ActorCriticPolicy, 
    MultiInputActorCriticPolicy,
)

MlpLstmPolicy = RecurrentActorCriticPolicy
CnnLstmPolicy = RecurrentActorCriticCnnPolicy
MultiInputLstmPolicy = RecurrentMultiInputActorCriticPolicy

MlPolicy = ActorCriticPolicy

register_policy("MlpLstmPolicy", RecurrentActorCriticPolicy)
register_policy("CnnLstmPolicy", RecurrentActorCriticCnnPolicy)
register_policy("MultiInputLstmPolicy", RecurrentMultiInputActorCriticPolicy)