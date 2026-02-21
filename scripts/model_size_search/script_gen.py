from pathlib import Path


if __name__ == "__main__":
    template_path = Path('template.sh')
    content = template_path.read_text()

    seeds = ['0', '1', '2']
    envs = [
        'antmaze-medium-navigate-v0',
        'antmaze-medium-stitch-v0',
        'antsoccer-arena-navigate-v0',
        'antsoccer-arena-stitch-v0',
        'humanoidmaze-medium-navigate-v0',
        'humanoidmaze-medium-stitch-v0',
        'pointmaze-medium-navigate-v0',
        'pointmaze-medium-stitch-v0'
    ]
    per_traj_samples_list = ['32', '512']
    num_layers_list = ['6', '9']

    output_path = Path("slurm_scripts")
    output_path.mkdir(exist_ok=True)

    for seed in seeds:
        for env in envs:
            _content = content.replace("{{seed}}", seed)
            _content = _content.replace("{{env}}", env)
            _content = _content.replace("{{per_traj_samples1}}", per_traj_samples_list[0])
            _content = _content.replace("{{per_traj_samples2}}", per_traj_samples_list[1])
            _content = _content.replace("{{num_hidden_layers1}}", num_layers_list[0])
            _content = _content.replace("{{num_hidden_layers2}}", num_layers_list[1])

            script_name = f"env_{env}_seed{seed}.sh"
            script_path = output_path / script_name
            script_path.write_text(_content)
