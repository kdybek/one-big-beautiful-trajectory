from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCBilinearPhi, GCBilinearValue, GCFlowMatchingActor, GCOneStepActor


class ACCRLAgent(flax.struct.PyTreeNode):
    """Action-Chunking Contrastive RL (ACCRL) agent.

    Extends CRL to output and evaluate action chunks (sequences of consecutive actions).
    The actor outputs a flat vector of chunk_length * action_dim, which is reshaped during
    evaluation into (chunk_length, action_dim) and executed sequentially.

    The critic's phi takes (s, flat_action_chunk) and psi takes (g); the contrastive loss and
    actor losses (AWR / DDPG+BC / FQL) operate identically to CRL on flattened chunks. AWR and
    FQL use a flow matching actor; FQL additionally trains a one-step student distilled from the
    flow teacher for fast inference.

    Decoupled Q-chunking (DQC): when ``decoupled_action_chunk_length`` (dacl) is set, a second
    critic phi_dqc(s, a_{1:dacl}) is distilled from the full-chunk critic via expectile
    regression (``kappa_dqc``), reusing its goal representation psi(g). The actor then outputs
    dacl-length chunks guided by this DQC critic.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def contrastive_loss(self, batch, grad_params, module_name='critic'):
        """Compute the contrastive value loss for the Q or V function."""
        batch_size = batch['observations'].shape[0]

        if module_name == 'critic':
            # Flatten action chunks: (B, chunk_length, action_dim) -> (B, chunk_length * action_dim)
            actions = batch['action_chunks'].reshape(batch_size, -1)
        else:
            actions = None
        v, phi, psi = self.network.select(module_name)(
            batch['observations'],
            batch['value_goals'],
            actions=actions,
            info=True,
            params=grad_params,
        )
        if len(phi.shape) == 2:  # Non-ensemble.
            phi = phi[None, ...]
            psi = psi[None, ...]
        logits = jnp.einsum('eik,ejk->ije', phi, psi) / jnp.sqrt(phi.shape[-1])
        # logits.shape is (B, B, e) with one term for positive pair and (B - 1) terms for negative pairs in each row.
        I = jnp.eye(batch_size)
        contrastive_loss = jax.vmap(
            lambda _logits: optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I),
            in_axes=-1,
            out_axes=-1,
        )(logits)
        contrastive_loss = jnp.mean(contrastive_loss)

        # Compute additional statistics.
        v = jnp.exp(v)
        logits = jnp.mean(logits, axis=-1)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return contrastive_loss, {
            'contrastive_loss': contrastive_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'binary_accuracy': jnp.mean((logits > 0) == I),
            'categorical_accuracy': jnp.mean(correct),
            'logits_pos': logits_pos,
            'logits_neg': logits_neg,
            'logits': logits.mean(),
        }

    def _dqc_q_values(self, observations, goals, actions):
        """Decoupled Q-chunking critic value: Q = phi_dqc(s, a_{1:dacl})^T psi(g) / sqrt(d).

        psi (the goal representation) is reused from the standard critic and detached, so
        only the DQC phi network is exercised here. Returns the (q1, q2) ensemble values.
        """
        # Dummy actions to satisfy the standard critic's phi input shape; psi ignores them.
        ref = jnp.zeros((*actions.shape[:-1], self.config['critic_action_dim']))
        _, _, psi = self.network.select('critic')(observations, goals, ref, info=True)
        psi = jax.lax.stop_gradient(psi)  # (2, B, d) for the ensemble critic.
        phi = self.network.select('dqc_phi')(observations, actions)  # (2, B, d)
        q = (phi * psi / jnp.sqrt(phi.shape[-1])).sum(axis=-1)  # (2, B)
        return q[0], q[1]

    def _q_for_actor(self, observations, goals, actions):
        """Q values used to guide the actor: DQC critic when enabled, else the standard critic."""
        if self.config['decoupled_action_chunk_length'] is None:
            return self.network.select('critic')(observations, goals, actions)
        return self._dqc_q_values(observations, goals, actions)

    def dqc_critic_loss(self, batch, grad_params):
        """Expectile regression distilling the full critic into the decoupled (DQC) critic.

        Trains phi_dqc(s, a_{1:dacl}) so that Q_dqc approximates an upper expectile of the
        full critic Q_full(s, a_{1:acl}, g):

            L = f^{kappa}_expectile( Q_full(s, a_{1:acl}, g) - Q_dqc(s, a_{1:dacl}, g) ),

        where Q_full and psi(g) are detached (only phi_dqc is trained). With kappa close to 1
        this pushes Q_dqc toward the max over the dropped continuation actions a_{dacl+1:acl}.
        """
        batch_size = batch['observations'].shape[0]
        dacl = self.config['decoupled_action_chunk_length']

        full_actions = batch['action_chunks'].reshape(batch_size, -1)
        dec_actions = batch['action_chunks'][:, :dacl].reshape(batch_size, -1)

        # Full critic value and goal representation (fixed targets; no gradient flows here).
        v_full, _, psi = self.network.select('critic')(
            batch['observations'], batch['value_goals'], full_actions, info=True,
        )
        v_full = jax.lax.stop_gradient(v_full)  # (2, B)
        psi = jax.lax.stop_gradient(psi)        # (2, B, d)

        phi_dec = self.network.select('dqc_phi')(
            batch['observations'], dec_actions, params=grad_params,
        )  # (2, B, d)
        q_dec = (phi_dec * psi / jnp.sqrt(phi_dec.shape[-1])).sum(axis=-1)  # (2, B)

        residual = v_full - q_dec
        kappa = self.config['kappa_dqc']
        weight = jnp.where(residual > 0, kappa, 1.0 - kappa)
        dqc_loss = (weight * residual ** 2).mean()

        return dqc_loss, {
            'dqc_loss': dqc_loss,
            'q_full_mean': v_full.mean(),
            'q_dec_mean': q_dec.mean(),
            'residual_mean': residual.mean(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss (AWR / FQL flow matching, or DDPG+BC Gaussian)."""
        # BC target: the full chunk, sliced to the actor's dacl actions under DQC.
        dacl = self.config['decoupled_action_chunk_length']
        chunks = batch['action_chunks'] if dacl is None else batch['action_chunks'][:, :dacl]
        flat_actions = chunks.reshape(chunks.shape[0], -1)

        if self.config['actor_loss'] in ('awr', 'fql'):
            # Flow matching velocity regression: trains the flow policy f_ξ ≈ π_β.
            rng, noise_rng, time_rng = jax.random.split(rng, 3)
            x_0 = jax.random.normal(noise_rng, flat_actions.shape)
            t = jax.random.uniform(time_rng, (flat_actions.shape[0],))

            # Interpolation: x_t = (1-t)*x_0 + t*x_1, target velocity: v* = x_1 - x_0.
            x_t = (1 - t[:, None]) * x_0 + t[:, None] * flat_actions
            v_target = flat_actions - x_0

            # Predicted velocity (gradient flows through actor params).
            v_pred = self.network.select('actor')(
                batch['observations'], batch['actor_goals'],
                actions=x_t, time=t,
                params=grad_params,
            )

            per_sample_fm = jnp.mean((v_pred - v_target) ** 2, axis=-1)
            bc_loss = per_sample_fm.mean()

            if self.config['actor_loss'] == 'awr':
                # AWR: advantage-weighted FM + BC regularization.
                # L = (exp(adv * alpha) * FM_loss).mean() + bc_coef * FM_loss.mean()
                v = self.network.select('value')(batch['observations'], batch['actor_goals'])
                q1, q2 = self._q_for_actor(batch['observations'], batch['actor_goals'], flat_actions)
                q = jnp.minimum(q1, q2)
                adv = q - v

                exp_a = jnp.exp(adv * self.config['alpha'])
                exp_a = jnp.minimum(exp_a, 100.0)

                adv_loss = (exp_a * per_sample_fm).mean()
                actor_loss = adv_loss + self.config['bc_coef'] * bc_loss

                return actor_loss, {
                    'actor_loss': actor_loss,
                    'adv_loss': adv_loss,
                    'bc_loss': bc_loss,
                    'adv': adv.mean(),
                    'fm_loss': bc_loss,
                }
            else:
                # fql: teacher BC loss + student (Q + distillation) loss.
                teacher_bc_loss = bc_loss

                # Student FQL loss: Q-guided + distillation from teacher.
                rng, student_noise_rng = jax.random.split(rng)
                z = jax.random.normal(student_noise_rng, flat_actions.shape)

                # Student one-step output (gradients flow through student params).
                a_student = self.network.select('student_actor')(
                    batch['observations'], batch['actor_goals'],
                    noise=z,
                    params=grad_params,
                )
                a_student = jnp.clip(a_student, -1, 1)

                # Teacher ODE output from the same noise z (no gradient).
                a_teacher = self._integrate_ode(batch['observations'], batch['actor_goals'], z)
                a_teacher = jax.lax.stop_gradient(a_teacher)

                # Q loss: maximize Q(s, a_student), normalized for scale invariance.
                q1, q2 = self._q_for_actor(batch['observations'], batch['actor_goals'], a_student)
                q = jnp.minimum(q1, q2)
                q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)

                # Distillation loss: MSE between student output and teacher ODE output.
                distill_loss = jnp.mean((a_student - a_teacher) ** 2, axis=-1).mean()

                student_loss = q_loss + self.config['alpha'] * distill_loss

                actor_loss = teacher_bc_loss + student_loss

                return actor_loss, {
                    'actor_loss': actor_loss,
                    'teacher_bc_loss': teacher_bc_loss,
                    'student_loss': student_loss,
                    'q_loss': q_loss,
                    'distill_loss': distill_loss,
                    'q_mean': q.mean(),
                    'q_abs_mean': jnp.abs(q).mean(),
                    'fm_loss': teacher_bc_loss,
                }

        elif self.config['actor_loss'] == 'ddpgbc':
            # DDPG+BC loss (unchanged Gaussian).
            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            q1, q2 = self._q_for_actor(batch['observations'], batch['actor_goals'], q_actions)
            q = jnp.minimum(q1, q2)

            # Normalize Q values by the absolute mean to make the loss scale invariant.
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            log_prob = dist.log_prob(flat_actions)

            bc_loss = -(self.config['alpha'] * log_prob).mean()

            actor_loss = q_loss + bc_loss

            return actor_loss, {
                'actor_loss': actor_loss,
                'q_loss': q_loss,
                'bc_loss': bc_loss,
                'q_mean': q.mean(),
                'q_abs_mean': jnp.abs(q).mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - flat_actions) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }
        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        critic_loss, critic_info = self.contrastive_loss(batch, grad_params, 'critic')
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        if self.config['decoupled_action_chunk_length'] is not None:
            dqc_loss, dqc_info = self.dqc_critic_loss(batch, grad_params)
            for k, v in dqc_info.items():
                info[f'dqc/{k}'] = v
        else:
            dqc_loss = 0.0

        if self.config['actor_loss'] == 'awr':
            value_loss, value_info = self.contrastive_loss(batch, grad_params, 'value')
            for k, v in value_info.items():
                info[f'value/{k}'] = v
        else:
            value_loss = 0.0

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + value_loss + actor_loss + dqc_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    def _integrate_ode(self, observations, goals, x):
        """Euler-integrate the flow ODE from initial noise `x` (B, action_dim) to actions.

        Operates on a flat batch; observations/goals must already have a leading batch dim.
        Returns actions of shape (B, action_dim) clipped to [-1, 1].
        """
        num_steps = self.config['num_flow_steps']
        dt = 1.0 / num_steps

        def step_fn(x, step_idx):
            t_batch = jnp.full((x.shape[0],), step_idx * dt)
            v = self.network.select('actor')(observations, goals, actions=x, time=t_batch)
            return x + v * dt, None

        x, _ = jax.lax.scan(step_fn, x, jnp.arange(num_steps))
        return jnp.clip(x, -1, 1)

    def _flow_sample(self, observations, goals, seed):
        """Sample actions from the flow policy via ODE integration from Gaussian noise.

        Handles batched (B, obs_dim) and unbatched (obs_dim,) inputs, returning actions with
        the matching leading shape ((B, action_dim) or (action_dim,)).
        """
        unbatched = (observations.ndim == 1)
        if unbatched:
            observations = observations[None]
            if goals is not None:
                goals = goals[None]

        x = jax.random.normal(seed, (observations.shape[0], self.config['total_action_dim']))
        x = self._integrate_ode(observations, goals, x)
        return x[0] if unbatched else x

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample action chunks from the actor.

        Returns:
            If action_chunk_length > 1: array of shape (*, chunk_length, per_step_action_dim).
            If action_chunk_length == 1: array of shape (*, action_dim) (same as CRL).
        """
        if self.config['actor_loss'] == 'ddpgbc':
            # Gaussian sampling (unchanged).
            dist = self.network.select('actor')(observations, goals, temperature=temperature)
            actions = dist.sample(seed=seed)
            actions = jnp.clip(actions, -1, 1)

        elif self.config['actor_loss'] == 'awr':
            # Flow matching ODE integration.
            actions = self._flow_sample(observations, goals, seed)

        elif self.config['actor_loss'] == 'fql':
            # FQL: student one-step generation.
            action_dim = self.config['total_action_dim']

            unbatched = (observations.ndim == 1)
            if unbatched:
                obs_expanded = observations[None]
                goals_expanded = goals[None] if goals is not None else None
            else:
                obs_expanded = observations
                goals_expanded = goals

            z = jax.random.normal(seed, (obs_expanded.shape[0], action_dim))
            actions = self.network.select('student_actor')(obs_expanded, goals_expanded, noise=z)
            actions = jnp.clip(actions, -1, 1)  # (B, action_dim)
            if unbatched:
                actions = actions[0]  # (action_dim,)
        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

        # Reshape flat output to (*, chunk_length, per_step_action_dim) for evaluation.
        # Under DQC the actor outputs decoupled_action_chunk_length actions.
        dacl = self.config['decoupled_action_chunk_length']
        chunk_length = dacl if dacl is not None else self.config['action_chunk_length']
        if chunk_length > 1:
            actions = actions.reshape(*actions.shape[:-1], chunk_length, -1)

        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new ACCRL agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. For action chunking, this should be the
                flattened action chunk of shape (batch, chunk_length * action_dim).
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        # action_dim is action_chunk_length * per_step_action_dim for the standard critic.
        action_dim = ex_actions.shape[-1]

        # Decoupled Q-chunking (DQC): the actor (and DQC critic) operate on the first
        # decoupled_action_chunk_length actions, while the standard critic keeps the full chunk.
        dacl = config['decoupled_action_chunk_length']
        if dacl is not None:
            assert dacl <= config['action_chunk_length'], (
                'decoupled_action_chunk_length must be <= action_chunk_length'
            )
            per_step_action_dim = action_dim // config['action_chunk_length']
            actor_action_dim = dacl * per_step_action_dim
            ex_actor_actions = ex_actions[:, :actor_action_dim]
        else:
            actor_action_dim = action_dim
            ex_actor_actions = ex_actions

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic_state'] = encoder_module()
            encoders['critic_goal'] = encoder_module()
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())
            if dacl is not None:
                encoders['dqc_state'] = encoder_module()
            if config['actor_loss'] == 'awr':
                encoders['value_state'] = encoder_module()
                encoders['value_goal'] = encoder_module()

        # Define value and actor networks.
        critic_def = GCBilinearValue(
            hidden_dims=config['value_hidden_dims'],
            latent_dim=config['latent_dim'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            value_exp=False,
            state_encoder=encoders.get('critic_state'),
            goal_encoder=encoders.get('critic_goal'),
        )

        if config['actor_loss'] == 'awr':
            # AWR requires a separate V network to compute advantages (Q - V).
            value_def = GCBilinearValue(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=False,
                value_exp=False,
                state_encoder=encoders.get('value_state'),
                goal_encoder=encoders.get('value_goal'),
            )

        if config['actor_loss'] in ('awr', 'fql'):
            # Flow matching actor.
            actor_def = GCFlowMatchingActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=actor_action_dim,
                layer_norm=config['layer_norm'],
                gc_encoder=encoders.get('actor'),
            )
            ex_time = jnp.zeros(ex_observations.shape[0])
            actor_init_args = (ex_observations, ex_goals, ex_actor_actions, ex_time)
        else:
            # Gaussian actor (ddpgbc).
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=actor_action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
            )
            actor_init_args = (ex_observations, ex_goals)

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, actor_init_args),
        )
        if dacl is not None:
            # DQC critic phi over (s, a_{1:dacl}); psi is reused from the standard critic.
            dqc_phi_def = GCBilinearPhi(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                state_encoder=encoders.get('dqc_state'),
            )
            network_info['dqc_phi'] = (dqc_phi_def, (ex_observations, ex_actor_actions))
        if config['actor_loss'] == 'awr':
            network_info.update(
                value=(value_def, (ex_observations, ex_goals)),
            )
        if config['actor_loss'] == 'fql':
            # Student one-step actor for FQL.
            if config['encoder'] is not None:
                encoders['student_actor'] = GCEncoder(concat_encoder=encoder_module())
            student_actor_def = GCOneStepActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=actor_action_dim,
                layer_norm=config['layer_norm'],
                gc_encoder=encoders.get('student_actor'),
            )
            ex_noise = jnp.zeros_like(ex_actor_actions)
            network_info['student_actor'] = (
                student_actor_def,
                (ex_observations, ex_goals, ex_noise),
            )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        # Store action dimensions in config. total_action_dim is the actor's flat output
        # dimension (used for flow sampling), which under DQC is decoupled_action_chunk_length
        # * per_step. critic_action_dim is the full chunk dimension of the standard critic.
        config_dict = dict(config)
        config_dict['total_action_dim'] = actor_action_dim
        config_dict['critic_action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config_dict))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='accrl',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims= 6 * (512,),  # Actor network hidden dimensions.
            value_hidden_dims= 6 * (512,),  # Value network hidden dimensions.
            latent_dim=512,  # Latent dimension for phi and psi.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.95,  # Discount factor.
            actor_loss='fql',  # Actor loss type ('awr', 'fql', or 'ddpgbc').
            alpha=0.1,  # Temperature in AWR, BC coefficient in DDPG+BC, or distillation weight in FQL.
            bc_coef=0.0,  # BC regularization coefficient for flow matching modes (awr).
            const_std=True,  # Whether to use constant standard deviation for the actor (ddpgbc only).
            discrete=False,  # Whether the action space is discrete (not supported with chunk_length > 1).
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Action chunking hyperparameters.
            action_chunk_length=1,  # Number of consecutive actions per chunk.
            # Decoupled Q-chunking (DQC). When set, the actor outputs a chunk of this length
            # (<= action_chunk_length) and is trained against a DQC critic distilled from the
            # standard (full-chunk) critic. None disables DQC.
            decoupled_action_chunk_length=ml_collections.config_dict.placeholder(int),
            kappa_dqc=0.9,  # Expectile parameter for the DQC critic distillation loss (DQC only).
            replan_length=ml_collections.config_dict.placeholder(int),  # Steps to execute per chunk before replanning (None = full chunk).
            # Flow matching hyperparameters.
            num_flow_steps=10,  # Number of Euler integration steps for flow matching ODE.
            # Dataset hyperparameters.
            dataset_class='ACGCDataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=False,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
