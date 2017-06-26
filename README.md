# gym-vrep

Makes a V-Rep simulation of the **Poppy Ergo Jr** robot available as OpenAI Gym environment.
 
Each of the environments moves the robot into a starting position before inserting the ball and starting the episode.

TODO: insert starting pose figure

The reward in each environment has two characteristics: metric and type. Metric specifies which value/relationship of the environment is observed and type specifies how the reward is returned to the agent. 

For the reward there are two types: maximum and current - "current" is directly corresponding to the current state of the system (e.g. if the ball is on the floor, and the metric is ball height then the reward is next to zero), and maximum means the reward at each point in time is the maximum reward that was achieved during the whole expisode for the given reward metric (e.g. if the ball is on the floor, and the metric is ball height then the reward is NOT zero, but instead the highest position that was achieved in this episode) 

### Environments:
- **ErgoBall-v0** (discontinued) - NO DYNAMICS, state consists only of current joint angles. 
  - *Reward type:* Maximum
  - *Reward metric:* Ball height 
- **ErgoBallDyn-v0** - Same as `ErgoBall-v0` but with current angular forces being part of the state.
  - *Reward type:* Maximum
  - *Reward metric:* Ball height 
- **ErgoBallDyn-v1** - Made for shorter episodes, because current ball height is rewarded, not maximum height.
  - *Reward type:* Current
  - *Reward metric:* Ball height 
- **ErgoBallThrow-v0** - New env to check if robot can be trained to throw ball, and not just lift it up high.
  - *Reward type:* Current
  - *Reward metric:* Z-distance between end effector (EE) and ball 
- **ErgoBallThrowVert-v0** - Similar to `ErgoBallThrow-v0` but now with focus on straight horizontal throws
  - *Reward type:* Current
  - *Reward metric:* Z-distance between EE and ball minus x/y-distance between EE and ball 
- **ErgoBallThrowVert-v1** - Same as `...-v0`, but with alternative reward type
  - *Reward type:* Mixed
  - *Reward metric:* Maximum(z-distance between EE and ball) minus current(x/y-distance between EE and ball) 
  