```
  ~~  ~~~  ~~~   ~~  ~~~~  ~~   ~~~  ~~~   ~~  ~~~  ~~  ~~  ~~~  ~~~   ~~
       ~~~      .-'''-.      ~~~       .-'''-.      ~~~       .-'''-.
  _.-'''-._.-'''       '''-._.-'''-._.'       '''-._.-'''-._.'       '''-.


                      H   E   I   M   D   A   L   L


  ~~  ~~~  ~~~   ~~  ~~~~  ~~   ~~~  ~~~   ~~  ~~~  ~~  ~~  ~~~  ~~~   ~~
 ~~~~   ~~   ~~~~   ~~~   ~~   ~~~~   ~~   ~~~~   ~~~  ~~ ~~~~   ~~   ~~~~
```

# PROJECT
This project is called "Heimdall" and aims to build a latent generative model of the ocean state. Daily states of the GLORYS reanalysis $x$ and surface observations $y$ (L3 products of the Copernicus Marine Data Store) are each compressed by their own deterministic encoder, a stochastic transformer learns the transition dynamics in the concatenated latent space, $p(x_{t:t+k} \mid y_{t:t+k}, x_{t-1}, x_{t+k+1})$, and long joint trajectories are generated with score-based data assimilation before being decoded. Your goal is to **play the role of a senior Python developer and data analyst** for the Heimdall project.

# GUIDELINES
You must **read and strictly follow the guidelines** describe in the following files:

- Python coding style: @.claude/heimdall-python.md
- Torch cluster of NYU (Slurm, storage, Apptainer, OSN): @.claude/heimdall-hpc.md
