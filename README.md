# Gym Gomoku Environment(Five-In-a-Row)
OpenAI Gym Env for game Gomoku(Five-In-a-Row, 五子棋, 五目並べ, omok, Gobang,...)
The game is played on a typical 19x19 or 15x15 go board. Black plays first and players 
alternate in placing a stone of their color on an empty intersection.
The winner is the first player to get an unbroken row of five stones horizontally, vertically, or diagonally.

# Additional function
Add duel mode to battle with each other models.
<pre>env = gym.make('Gomoku9x9duel-v0')</pre>
In duel mode, you can know who win the game by reward.
If win black stone, reward is 1, but win white stone, reward is -1.

# Acks
This project was started by Xichen. His fork can be found at https://github.com/rockingdingo/gym-gomoku.

# Demo
![gym_gomoku_demo](demo/gym_gomoku_demo.gif)

# Installation
Github
```bash
git clone https://github.com/onetop21/gym-gomoku.git
cd gym-gomoku
pip install -e .
```

# Quick example
```python

import gym
import gym_gomoku
env = gym.make('Gomoku19x19-v0') # default 'beginner' level opponent policy

env.reset()
env.render()
env.step(15) # place a single stone, black color first

# play a game
env.reset()
for _ in range(20):
    action = env.action_space.sample() # sample without replacement
    observation, reward, done, info = env.step(action)
    env.render()
    if done:
        print ("Game is Over")
        break
```

