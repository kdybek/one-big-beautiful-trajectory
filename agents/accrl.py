from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCBilinearValue, GCFlowMatchingActor, GCOneStepActor


class ACCRLAgent(flax.struct.PyTreeNode):
    """Action-Chunking Contrastive RL (ACCRL) agent.

    Extends CRL to output and evaluate action chunks (sequences of consecutive actions).
    The actor outputs a flat vector of chunk_length * action_dim, which is reshaped during
    evaluation into (chunk_length, action_dim) and executed sequentially.

    The critic's phi takes (s, flat_action_chunk) and psi takes (g). The contrastive loss
    and actor losses (AWR / DDPG+BC / best-of-N / FQL) operate identically to CRL, just on
    flattened action chunks. AWR, best-of-N, and FQL modes use a flow matching actor instead
    of a Gaussian actor. FQL additionally trains a one-step student policy distilled from the
    flow teacher and guided by the CRL critic, enabling fast best-of-N inference.
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

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss (AWR / best-of-N / FQL flow matching, or DDPG+BC Gaussian)."""
        # Flatten action chunks for all actor loss computations.
        flat_actions = batch['action_chunks'].reshape(batch['action_chunks'].shape[0], -1)

        if self.config['actor_loss'] in ('awr', 'bestofn', 'fql'):
            # Flow matching velocity regression (behavioral cloning component).
            # This trains f_ξ ≈ π_β: the flow policy to capture the behavior distribution.
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
            # Unweighted BC loss: ensures the flow policy faithfully captures π_β.
            bc_loss = per_sample_fm.mean()

            if self.config['actor_loss'] == 'awr':
                # AWR: advantage-weighted FM + explicit BC regularization.
                # L = (exp(adv * alpha) * FM_loss).mean() + bc_coef * FM_loss.mean()
                # The first term biases toward high-advantage actions;
                # the second term (behavior constraint) keeps the policy close to π_β.
                v = self.network.select('value')(batch['observations'], batch['actor_goals'])
                q1, q2 = self.network.select('critic')(batch['observations'], batch['actor_goals'], flat_actions)
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
            elif self.config['actor_loss'] == 'bestofn':
                # bestofn: pure BC via flow matching. Q is only used at inference
                # for best-of-N selection (implicit KL constraint, Eq. 10 in Q-chunking).
                actor_loss = bc_loss

                return actor_loss, {
                    'actor_loss': actor_loss,
                    'bc_loss': bc_loss,
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
                a_teacher = self._flow_sample_from_noise(
                    batch['observations'], batch['actor_goals'], z,
                )
                a_teacher = jax.lax.stop_gradient(a_teacher)

                # Q loss: maximize Q(s, a_student), normalized for scale invariance.
                q1, q2 = self.network.select('critic')(
                    batch['observations'], batch['actor_goals'], a_student,
                )
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
            q1, q2 = self.network.select('critic')(batch['observations'], batch['actor_goals'], q_actions)
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

        loss = critic_loss + value_loss + actor_loss
        return loss, info

    @jax.jit
    def _action_sensitivity_metrics(self, batch, rng, n_samples=1000):
        """Measure Q-value statistics and critic sensitivity to actions.

        Computes:
        - Var/mean of Q over random uniform actions at a fixed (s, g).
        - ||grad_a Q(s,a)|| / sqrt(action_dim) at the current policy action (flow sample).
        - ||grad_a Q(s,a)|| / sqrt(action_dim) averaged over batch actions from the replay buffer.
        """
        s = batch['observations'][0:1]       # (1, obs_dim)
        g = batch['value_goals'][0:1]        # (1, goal_dim)
        action_dim = self.config['total_action_dim']

        # --- Q variance over random uniform actions ---
        rng, uniform_rng = jax.random.split(rng)
        random_actions = jax.random.uniform(uniform_rng, (n_samples, action_dim), minval=-1.0, maxval=1.0)
        s_rep = jnp.repeat(s, n_samples, axis=0)
        g_rep = jnp.repeat(g, n_samples, axis=0)
        q1, q2 = self.network.select('critic')(s_rep, g_rep, random_actions)
        q = jnp.minimum(q1, q2)

        # --- grad_a Q at policy action (student actor, consistent with evaluation) ---
        rng, noise_rng = jax.random.split(rng)
        z = jax.random.normal(noise_rng, (1, action_dim))
        policy_action = self.network.select('student_actor')(s, g, noise=z)
        policy_action = jnp.clip(policy_action, -1, 1)  # (1, action_dim)

        def q_fn_policy(a):
            q1, q2 = self.network.select('critic')(s, g, a)
            return jnp.minimum(q1, q2).squeeze()

        grad_policy = jax.grad(q_fn_policy)(policy_action)   # (1, action_dim)
        grad_norm_policy = jnp.linalg.norm(grad_policy) / jnp.sqrt(action_dim)

        # --- grad_a Q averaged over batch actions ---
        flat_actions = batch['action_chunks'].reshape(batch['action_chunks'].shape[0], -1)

        def q_fn_single(s_i, g_i, a_i):
            q1, q2 = self.network.select('critic')(s_i[None], g_i[None], a_i[None])
            return jnp.minimum(q1, q2).squeeze()

        grad_fn = jax.grad(q_fn_single, argnums=2)
        grad_batch = jax.vmap(grad_fn)(batch['observations'], batch['value_goals'], flat_actions)
        grad_norm_batch = jnp.mean(jnp.linalg.norm(grad_batch, axis=-1)) / jnp.sqrt(action_dim)

        return {
            'critic/action_q_var': jnp.var(q),
            'critic/action_q_mean': jnp.mean(q),
            'critic/grad_norm_policy_action': grad_norm_policy,
            'critic/grad_norm_batch_action': grad_norm_batch,
        }

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    def _flow_sample(self, observations, goals, seed, num_samples=1):
        """Generate actions via Euler ODE integration of the flow matching velocity field.

        Args:
            observations: Observations of shape (B, obs_dim) or (obs_dim,).
            goals: Goals (optional), same leading shape as observations.
            seed: PRNG key.
            num_samples: Number of samples to generate per observation. If > 1,
                observations/goals are replicated along a leading axis.

        Returns:
            Unbatched input → (num_samples, action_dim) if num_samples > 1, else (action_dim,).
            Batched input   → (num_samples, B, action_dim) if num_samples > 1, else (B, action_dim).
        """
        action_dim = self.config['total_action_dim']
        num_steps = self.config['num_flow_steps']
        dt = 1.0 / num_steps

        # Normalize to always have a batch dimension so shape[0] is always the batch size.
        unbatched = (observations.ndim == 1)
        if unbatched:
            observations = observations[None]
            if goals is not None:
                goals = goals[None]
        batch_size = observations.shape[0]

        if num_samples > 1:
            # Replicate obs/goals: (N*B, obs_dim) for batched velocity evaluation.
            obs_rep = jnp.broadcast_to(
                observations[None], (num_samples, batch_size, *observations.shape[1:])
            )
            obs_rep = obs_rep.reshape(num_samples * batch_size, *observations.shape[1:])
            if goals is not None:
                goals_rep = jnp.broadcast_to(
                    goals[None], (num_samples, batch_size, *goals.shape[1:])
                )
                goals_rep = goals_rep.reshape(num_samples * batch_size, *goals.shape[1:])
            else:
                goals_rep = None
            x = jax.random.normal(seed, (num_samples * batch_size, action_dim))
        else:
            obs_rep = observations
            goals_rep = goals
            x = jax.random.normal(seed, (batch_size, action_dim))

        # Euler integration via jax.lax.scan.
        def step_fn(x, step_idx):
            t = step_idx * dt
            t_batch = jnp.full((x.shape[0],), t)
            v = self.network.select('actor')(obs_rep, goals_rep, actions=x, time=t_batch)
            x = x + v * dt
            return x, None

        x, _ = jax.lax.scan(step_fn, x, jnp.arange(num_steps))
        x = jnp.clip(x, -1, 1)

        if num_samples > 1:
            x = x.reshape(num_samples, batch_size, action_dim)
            if unbatched:
                x = x[:, 0, :]  # (N, action_dim)
        else:
            if unbatched:
                x = x[0]  # (action_dim,)

        return x

    def _flow_sample_from_noise(self, observations, goals, noise):
        """Generate actions via Euler ODE integration starting from given noise.

        Unlike _flow_sample, this takes pre-generated noise (not a seed) and
        always operates on a flat batch. Used during FQL training so teacher
        and student share the same noise vector z.

        Args:
            observations: Observations of shape (B, obs_dim).
            goals: Goals of shape (B, goal_dim) or None.
            noise: Pre-generated noise of shape (B, action_dim).

        Returns:
            Actions of shape (B, action_dim).
        """
        num_steps = self.config['num_flow_steps']
        dt = 1.0 / num_steps

        x = noise

        def step_fn(x, step_idx):
            t = step_idx * dt
            t_batch = jnp.full((x.shape[0],), t)
            v = self.network.select('actor')(observations, goals, actions=x, time=t_batch)
            x = x + v * dt
            return x, None

        x, _ = jax.lax.scan(step_fn, x, jnp.arange(num_steps))
        x = jnp.clip(x, -1, 1)

        return x

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
        q_info = None  # set in bestofn branch; triggers tuple return

        if self.config['actor_loss'] == 'ddpgbc':
            # Gaussian sampling (unchanged).
            dist = self.network.select('actor')(observations, goals, temperature=temperature)
            actions = dist.sample(seed=seed)
            actions = jnp.clip(actions, -1, 1)

        elif self.config['actor_loss'] == 'awr':
            # Flow matching ODE integration (single sample).
            # temperature scales initial noise (0 = deterministic).
            actions = self._flow_sample(observations, goals, seed, num_samples=1)

        elif self.config['actor_loss'] == 'bestofn':
            # Flow matching ODE + best-of-N Q-guided selection.
            N = self.config['best_of_n']
            # Generate N candidate action chunks.
            # Unbatched obs → candidates: (N, action_dim)
            # Batched obs   → candidates: (N, B, action_dim)
            candidates = self._flow_sample(observations, goals, seed, num_samples=N)

            # Ensure candidates has batch dimension for vmap (N=1 unbatched case returns no batch dim)
            if observations.ndim == 1 and candidates.ndim == 1:
                candidates = candidates[None]  # (action_dim,) -> (1, action_dim)

            # Replicate obs/goals N times for parallel Q evaluation.
            # Unbatched: obs_rep (N, obs_dim); Batched: obs_rep (N, B, obs_dim)
            obs_rep = jnp.broadcast_to(observations[None], (N, *observations.shape))
            if goals is not None:
                goals_rep = jnp.broadcast_to(goals[None], (N, *goals.shape))
            else:
                goals_rep = None

            # Compute Q values for all N candidates via vmap over N.
            def eval_q(candidate, obs, gls):
                q1, q2 = self.network.select('critic')(obs, gls, candidate)
                return jnp.minimum(q1, q2)

            # Unbatched: q_vals (N,); Batched: q_vals (N, B)
            q_vals = jax.vmap(eval_q)(candidates, obs_rep, goals_rep)

            q_info = {
                'bestofn_q_var': jnp.var(q_vals),
                'bestofn_q_mean': jnp.mean(q_vals),
            }

            # Select best candidate: argmax over N dimension.
            best_idx = jnp.argmax(q_vals, axis=0)  # scalar or (B,)
            if observations.ndim == 1:
                # Unbatched: candidates (N, action_dim), best_idx is scalar.
                actions = candidates[best_idx]  # (action_dim,)
            else:
                # Batched: candidates (N, B, action_dim), best_idx is (B,).
                actions = candidates[best_idx, jnp.arange(observations.shape[0])]  # (B, action_dim)

        elif self.config['actor_loss'] == 'fql':
            # FQL: student one-step generation + best-of-N Q-guided selection.
            N = self.config['best_of_n']
            action_dim = self.config['total_action_dim']

            unbatched = (observations.ndim == 1)
            if unbatched:
                obs_expanded = observations[None]
                goals_expanded = goals[None] if goals is not None else None
            else:
                obs_expanded = observations
                goals_expanded = goals
            batch_size = obs_expanded.shape[0]

            # Generate N noise vectors and replicate obs/goals.
            z = jax.random.normal(seed, (N, batch_size, action_dim))
            obs_rep = jnp.broadcast_to(
                obs_expanded[None], (N, batch_size, *obs_expanded.shape[1:])
            )
            if goals_expanded is not None:
                goals_rep = jnp.broadcast_to(
                    goals_expanded[None], (N, batch_size, *goals_expanded.shape[1:])
                )
            else:
                goals_rep = None

            # Student one-step forward pass via vmap over N dimension.
            def student_forward(obs, gls, noise):
                a = self.network.select('student_actor')(obs, gls, noise=noise)
                return jnp.clip(a, -1, 1)

            candidates = jax.vmap(student_forward)(obs_rep, goals_rep, z)  # (N, B, action_dim)

            # Q evaluation for best-of-N selection.
            def eval_q(candidate, obs, gls):
                q1, q2 = self.network.select('critic')(obs, gls, candidate)
                return jnp.minimum(q1, q2)

            q_vals = jax.vmap(eval_q)(candidates, obs_rep, goals_rep)  # (N, B)

            best_idx = jnp.argmax(q_vals, axis=0)
            if unbatched:
                best_idx = best_idx.squeeze()  # (1,) -> scalar
                actions = candidates[best_idx, 0]  # (action_dim,)
            else:
                actions = candidates[best_idx, jnp.arange(batch_size)]  # (B, action_dim)
        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

        # Reshape flat output to (*, chunk_length, per_step_action_dim) for evaluation.
        chunk_length = self.config['action_chunk_length']
        if chunk_length > 1:
            actions = actions.reshape(*actions.shape[:-1], chunk_length, -1)

        if q_info is not None:
            return actions, q_info
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
        # action_dim is chunk_length * per_step_action_dim for ACCRL.
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic_state'] = encoder_module()
            encoders['critic_goal'] = encoder_module()
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())
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

        if config['actor_loss'] in ('awr', 'bestofn', 'fql'):
            # Flow matching actor.
            actor_def = GCFlowMatchingActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                layer_norm=config['layer_norm'],
                gc_encoder=encoders.get('actor'),
            )
            ex_time = jnp.zeros(ex_observations.shape[0])
            actor_init_args = (ex_observations, ex_goals, ex_actions, ex_time)
        else:
            # Gaussian actor (ddpgbc).
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
            )
            actor_init_args = (ex_observations, ex_goals)

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, actor_init_args),
        )
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
                action_dim=action_dim,
                layer_norm=config['layer_norm'],
                gc_encoder=encoders.get('student_actor'),
            )
            ex_noise = jnp.zeros_like(ex_actions)
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

        # Store total_action_dim in config for flow sampling.
        config_dict = dict(config)
        config_dict['total_action_dim'] = action_dim

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
            discount=0.99,  # Discount factor.
            actor_loss='bestofn',  # Actor loss type ('awr', 'bestofn', 'fql', or 'ddpgbc').
            alpha=0.1,  # Temperature in AWR, BC coefficient in DDPG+BC, or distillation weight in FQL.
            bc_coef=0.0,  # BC regularization coefficient for flow matching modes (awr).
            const_std=True,  # Whether to use constant standard deviation for the actor (ddpgbc only).
            discrete=False,  # Whether the action space is discrete (not supported with chunk_length > 1).
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Action chunking hyperparameters.
            action_chunk_length=5,  # Number of consecutive actions per chunk.
            replan_length=ml_collections.config_dict.placeholder(int),  # Steps to execute per chunk before replanning (None = full chunk).
            shift_goals=True,  # Whether to shift goal sampling (action_chunk_length - 1) ahead of the current state.
            # Flow matching hyperparameters.
            num_flow_steps=10,  # Number of Euler integration steps for flow matching ODE.
            best_of_n=1,  # Number of candidates for best-of-N selection (bestofn and fql modes).
            # Dataset hyperparameters.
            dataset_class='ADGCDataset',  # Dataset class name.
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
