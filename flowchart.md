Action: 基于末端坐标系
Transform_Action: 基于基座坐标


```
[Model]:Action  ->  transform      ->  transformed_action                -> Robot

Action          <-  inv_transform  <-  [Space Mouse]:transformed_action  -> Robot
      ⬇️
data collect
```

## data collect

```bash
actions = np.zeros(env.action_space.sample().shape) 
next_obs, rew, done, truncated, info = env.step(actions)
returns += rew
if "intervene_action" in info:
    actions = info["intervene_action"]
transition = copy.deepcopy(
    dict(
        observations=obs,
        actions=actions,
        next_observations=next_obs,
        rewards=rew,
        masks=1.0 - done,
        dones=done,
        infos=info,
    )
)
trajectory.append(transition)
```

## env step

```bash
[GripperPenaltyWrapper].step(action)
└── [ChunkingWrapper].step(action)
    └── [SERLObsWrapper].step(action)
        └── [Quat2EulerWrapper].step(action)
            └── [RelativeFrame].step(action)
                ├── action = transform_action(action)
                └── [SpacemouseIntervention].step(action)
                    ├── new_action = [SpacemouseIntervention].action()
                    ├── [Robot].step(action)
                    └── info["intervene_action"] = new_action
                ├── info["intervene_action"] = transform_action_inv(info["intervene_action"])
                └── transformed_obs = transform_observation(obs)
            └── [Quat2EulerWrapper].observation(obs)
        └── [SERLObsWrapper].observation(obs)
    └── action = info["intervene_action"]

```