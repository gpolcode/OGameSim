## Setup rocm pytorch on windows
https://www.reddit.com/r/ROCm/comments/1nffbzt/install_rocm_pytorch_on_windows_with_amd_radeon/

Run https://docs.cleanrl.dev/get-started/installation/ but with "Run-UV pip install ."
Run the first script again

## Run
."C:\Users\Elsahr\rocm-pytorch\.venv\Scripts\activate.ps1"

https://gymnasium.farama.org/introduction/basic_usage/
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
https://github.com/openai/gym/wiki/Table-of-environments
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
https://docs.cleanrl.dev/advanced/resume-training/?h=save#resume-training_1

SPS Performance per envs:
1000: 43000
2000: 46000
4000: 60000
8000: 60000
16000: 56000
64000: 22000