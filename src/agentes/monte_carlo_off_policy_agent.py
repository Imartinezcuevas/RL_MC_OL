"""
Module: agentes/monte_carlo_off_policy.py
Description: Implementación de la clase base para agentes de Monte Carlo Off-Policy.

Author: Iván Martínez Cuevas
Email: ivan.martinezc@um.es
Date: 2025/02/24

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from agentes.monte_carlo_agent import MonteCarloAgent
import numpy as np
from typing import Any
from politicas import EpsilonGreedyPolicy

class MonteCarloOffPolicyAgent(MonteCarloAgent):
    """
    Agente de Monte Carlo off-policy usando importance sampling
    """
    
    def _init_algorithm_params(self, **kwargs):
        """
        Inicializa parámetros específicos para Monte Carlo off-policy
        
        Args:
            **kwargs: Parámetros adicionales
        """
        super()._init_algorithm_params(**kwargs)
        
        # Política objetivo (generalmente greedy)
        self.target_policy = kwargs.get('target_policy', EpsilonGreedyPolicy(self.action_space, epsilon=0.1))
        
        # Política de comportamiento (para explorar)
        self.behavior_policy = self.policy
        
        # Indica si se usa importance sampling ponderado
        self.weighted_is = kwargs.get('weighted_is', True)
        
        # Para importance sampling ponderado
        self.C = np.zeros((self.n_states, self.n_actions))
    
    def get_action(self, state: Any) -> int:
        """
        Obtiene una acción usando la política de comportamiento
        
        Args:
            state: Estado actual
            
        Returns:
            Acción seleccionada
        """
        return self.behavior_policy.select_action(state, self.Q)
    
    def _process_episode(self):
        """
        Procesa el episodio completo según el algoritmo de Monte Carlo off-policy
        """
        if not self.episode_buffer:
            return
            
        # Calcula retorno y peso de importance sampling
        G = 0
        W = 1.0  # Peso de importance sampling
        
        # Para first-visit Monte Carlo
        if self.first_visit:
            visited = set()
        
        # Recorre el episodio en orden inverso
        for t in range(len(self.episode_buffer) - 1, -1, -1):
            state, action, reward = self.episode_buffer[t]
            
            # Calcula el retorno
            G = self.gamma * G + reward
            
            # Para first-visit Monte Carlo
            if self.first_visit:
                if (state, action) in visited:
                    continue
                visited.add((state, action))
            
            # Actualiza C para importance sampling ponderado
            if self.weighted_is:
                self.C[state, action] += W
            
            # Actualiza Q con importance sampling
            if self.weighted_is and self.C[state, action] > 0:
                self.Q[state, action] += W / self.C[state, action] * (G - self.Q[state, action])
            else:
                # Importance sampling ordinario
                alpha = 1.0 / self.visit_counts[state, action] if self.visit_counts[state, action] > 0 else 0.1
                self.Q[state, action] += W * alpha * (G - self.Q[state, action])
            
            # Actualiza el peso del importance sampling
            target_probs = self.target_policy.get_action_probabilities(state, self.Q)
            behavior_probs = self.behavior_policy.get_action_probabilities(state, self.Q)
            
            # Si la acción tiene probabilidad cero en la política de comportamiento, termina
            if behavior_probs[action] == 0:
                break
                
            # Actualiza el peso usando el ratio de importance sampling
            W *= target_probs[action] / behavior_probs[action]
            
            # Si el peso llega a cero, no afectará futuras actualizaciones
            if W == 0:
                break
            
            # Incrementa el contador de visitas
            self.visit_counts[state, action] += 1