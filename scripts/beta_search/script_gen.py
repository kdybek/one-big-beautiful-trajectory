from pathlib import Path


if __name__ == "__main__":
    template_path = Path('template.sh')
    content = template_path.read_text()

    seeds = ['0', '1', '2']

    envs = [
        {
            'name': 'antmaze-medium-navigate-v0',
            'alpha': '0.1',
            'actor_p_randomgoal': '0.0',
            'actor_p_trajgoal': '1.0',
            'discount': '0.99',
        },
        {
            'name': 'antmaze-medium-stitch-v0',
            'alpha': '0.1',
            'actor_p_randomgoal': '0.5',
            'actor_p_trajgoal': '0.5',
            'discount': '0.99',
        },
        {
            'name': 'antsoccer-arena-navigate-v0',
            'alpha': '0.3',
            'actor_p_randomgoal': '0.0',
            'actor_p_trajgoal': '1.0',
            'discount': '0.99',
        },
        {
            'name': 'antsoccer-arena-stitch-v0',
            'alpha': '0.3',
            'actor_p_randomgoal': '0.5',
            'actor_p_trajgoal': '0.5',
            'discount': '0.99',
        },
        {
            'name': 'humanoidmaze-medium-navigate-v0',
            'alpha': '0.1',
            'actor_p_randomgoal': '0.0',
            'actor_p_trajgoal': '1.0',
            'discount': '0.995',
        },
        {
            'name': 'humanoidmaze-medium-stitch-v0',
            'alpha': '0.1',
            'actor_p_randomgoal': '0.5',
            'actor_p_trajgoal': '0.5',
            'discount': '0.995',
        },
        {
            'name': 'pointmaze-medium-navigate-v0',
            'alpha': '0.03',
            'actor_p_randomgoal': '0.0',
            'actor_p_trajgoal': '1.0',
            'discount': '0.99',
        },
        {
            'name': 'pointmaze-medium-stitch-v0',
            'alpha': '0.03',
            'actor_p_randomgoal': '0.5',
            'actor_p_trajgoal': '0.5',
            'discount': '0.99',
        },
    ]

    betas = ['0.0', '0.01', '0.001']

    output_path = Path("slurm_scripts")
    output_path.mkdir(exist_ok=True)

    for seed in seeds:
        for env in envs:
            _content = content.replace("{{seed}}", seed)
            _content = _content.replace("{{env}}", env['name'])
            _content = _content.replace("{{alpha}}", env['alpha'])
            _content = _content.replace("{{actor_p_randomgoal}}", env['actor_p_randomgoal'])
            _content = _content.replace("{{actor_p_trajgoal}}", env['actor_p_trajgoal'])
            _content = _content.replace("{{discount}}", env['discount'])
            _content = _content.replace(
                "{{beta1}}", betas[0])
            _content = _content.replace(
                "{{beta2}}", betas[1])
            _content = _content.replace(
                "{{beta3}}", betas[2])

            script_name = f"env_{env['name']}_seed{seed}.sh"
            script_path = output_path / script_name
            script_path.write_text(_content)
