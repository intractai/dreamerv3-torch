import gym
import numpy as np
import os


WORDS_PATH = './list_of_words.txt'
DEFAULT_COMMAND_TOKEN = '<|User|>'
DEFAULT_RESPONSE_TOKEN = '<|Assistant|>'


def load_text(file_path):
  """Loads a list of lines of text from the given path."""
  with open(file_path, 'r') as f:
    return f.read().split('\n')


class ToyTextEnv:
    metadata = {}

    def __init__(self, task, tokenizer, n_acts, goal_words=None, seed=0):
        self._task = task
        self._tokenizer = tokenizer
        self.n_acts = n_acts
        self._env = None
        self._done = True
        self._goal_words = goal_words or load_text(
        os.path.join(os.path.dirname(__file__), WORDS_PATH))
        self._obs_size = 64
        self._half_obs_size = int(self._obs_size / 2)
        self._char_limit = 512
        self._text_hist = '' # What the agent has typed in the current episode
        self._n_correct_tokens = 0 # How many correct tokens in a row the agent has outputted
        self._command_token = DEFAULT_COMMAND_TOKEN
        self._response_token = DEFAULT_RESPONSE_TOKEN

        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        spaces = {
            'text': gym.spaces.Box(0, self._tokenizer.vocab_size, (self._obs_size,), dtype=np.int32),
            'is_first': gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
            'is_last': gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
            'is_terminal': gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
            'reward': gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
        }
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        # All actions are specified in terms of tokens, not characters
        act_space = gym.spaces.Discrete(self.n_acts)
        act_space.discrete = True
        return act_space
  
    def get_goal(self):
        """
        Get a random word to create the tokenized goal string.
        Returns tokenized goal string and tokenize goal word.
        """
        goal_word = np.random.choice(self._goal_words)
        tokenized_goal_word = self._tokenizer(goal_word, padding=False, add_special_tokens=False).input_ids

        goal_text = self._command_token + f'write "{goal_word}"\n{self._response_token}'
        tokenized_goal_str = self._tokenizer(goal_text, padding=False, add_special_tokens=False).input_ids
        return tokenized_goal_str, tokenized_goal_word

    def reset(self):
        """Reset the environment and choose a new goal word."""
        self._done = False
        self._text_hist = ''
        self._tokenized_goal_str, self._tokenized_goal_word = self.get_goal()
        self._n_goal_tokens = len(self._tokenized_goal_word)
        self._n_correct_tokens = 0
        return self._obs(0.0, is_first=True)[0]
    
    def step(self, action):
        """
        If the agent types a correct token, give reward of 1.
        If it is the last token of the reward, end the episode.
        If it is an incorrect token, give a negative reward to cancel
        out anything the agent has earned so far in the episode.
        """
        if self._done:
            return self.reset()
        
        action_text = self._tokenizer.decode(action)
        self._text_hist += action_text
        
        # If agent enters correct token
        if action == self._tokenized_goal_word[self._n_correct_tokens]:
            reward = np.float32(1)
            self._n_correct_tokens += 1
            done = self._n_correct_tokens == self._n_goal_tokens

        # If agent enters incorrect token
        else:
            reward = np.float32(-self._n_correct_tokens)
            self._n_correct_tokens = 0
            done = False

        return self._obs(reward, is_last=done, is_terminal=done)

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
        # Tokenize and put together the full observation
        tokenized_text = self._tokenizer(
            self._text_hist, padding=False, add_special_tokens=False).input_ids
        obs_ids = [self._tokenizer.bos_token_id] \
            + self._tokenized_goal_str[-self._half_obs_size+1:] \
            + tokenized_text[-self._half_obs_size:]
        # Pad to the max length
        obs_ids = obs_ids + [self._tokenizer.pad_token_id] * (self._obs_size - len(obs_ids))

        return (
            dict(
                text=np.array(obs_ids, dtype=np.int32),
                reward=reward,
                is_first=is_first,
                is_last=is_last,
                is_terminal=is_terminal
            ),
            reward,
            is_last,
            {},
        )

    def render(self):
        return self._env.render()


### Test the environment in the main function ###

if __name__ == '__main__':
    # pass
    from transformers import AutoTokenizer

    model_name = 'deepseek-ai/deepseek-coder-1.3b-instruct'
    os.environ['TOKENIZERS_PARALLELISM'] = \
        os.environ.get('TOKENIZERS_PARALLELISM', 'true')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    env = ToyTextEnv(task='reward', tokenizer=tokenizer)
    
    # Test correct input
    obs = env.reset()
    print('Obs:', tokenizer.decode(obs['text'], skip_special_tokens=True))
    for action in env._tokenized_goal_word:
        obs, reward, done, info = env.step(action)
        print('Obs:', tokenizer.decode(obs['text'], skip_special_tokens=True))
        print(f'Reward: {reward} | Done: {done}')
    
    # Test intially incorrect but then correct input
    obs = env.reset()
    print('Obs:', tokenizer.decode(obs['text'], skip_special_tokens=True))
    for _ in range(2):
        action = np.random.randint(0, tokenizer.vocab_size)
        obs, reward, done, info = env.step(action)
        print('Obs:', tokenizer.decode(obs['text'], skip_special_tokens=True))
        print(f'Reward: {reward} | Done: {done}')

    for action in env._tokenized_goal_word:
        obs, reward, done, info = env.step(action)
        print('Obs:', tokenizer.decode(obs['text'], skip_special_tokens=True))
        print(f'Reward: {reward} | Done: {done}')
