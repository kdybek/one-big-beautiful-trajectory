import numpy as np
import jax
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import wandb


def plot_crl_pca(agent, dataset, n_traj, n_states, logger):
    """Plot PCA of the agent's latent representations for a number of trajectories.

    Args:
        agent: Agent with an encoder to obtain latent representations.
        dataset: Dataset to sample trajectories from.
        num_traj: Number of trajectories to plot.
    """
    phis = []
    psis = []
    traj_ids = []
    types = []

    trajs = dataset.sample_trajectories(n_traj)

    for traj_id, traj in enumerate(trajs):
        first_idx = max(len(traj['observations']) - n_states - 1, 0)
        observations = traj['observations'][first_idx:-1]
        goals = traj['observations'][first_idx + 1:]
        actions = traj['actions'][first_idx:-1]

        v, phi, psi = agent.network.select('critic')(
            observations=observations,
            goals=goals,
            actions=actions,
            info=True,
        )

        phi = np.array(jax.device_get(phi))
        psi = np.array(jax.device_get(psi))

        # Average over ensemble members
        if phi.ndim == 3:
            phi = phi.mean(axis=0)  # (T, D)
            psi = psi.mean(axis=0)  # (T, D)

        psi = psi[-1].reshape(1, -1)  # (1, D)

        phis.append(phi)
        psis.append(psi)

        traj_ids.extend([traj_id] * len(phi))
        traj_ids.append(traj_id)

        types.extend(["state"] * len(phi))
        types.append("goal")

    all_reps = np.concatenate(phis + psis, axis=0)

    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(all_reps)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)

    latents_2d = np.asarray(latents_2d)

    for traj_id in range(n_traj):
        idx = np.array(traj_ids) == traj_id

        states = idx & (np.array(types) == "state")
        goals = idx & (np.array(types) == "goal")

        ax.scatter(
            latents_2d[states, 0],
            latents_2d[states, 1],
            label=f"traj {traj_id}",
            alpha=0.7,
            marker="o",
        )

        ax.scatter(
            latents_2d[goals, 0],
            latents_2d[goals, 1],
            label=f"goal {traj_id}",
            alpha=1.0,
            marker="*",
        )

    ax.set_title("CRL Latent PCA")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(loc="best", fontsize=8)
    ax.axis("equal")
    ax.grid(True)

    logger.log({
        "visualizations/pca": wandb.Image(fig)
    })

    plt.close(fig)
