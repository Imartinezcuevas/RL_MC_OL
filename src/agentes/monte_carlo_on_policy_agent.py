"""
Module: agentes/monte_carlo_on_policy.py
Description: Implementación de la clase base para agentes de Monte Carlo On-Policy.

Author: Iván Martínez Cuevas
Email: ivan.martinezc@um.es
Date: 2025/02/24

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from agentes.monte_carlo_agent import MonteCarloAgent
import numpy as np

class MonteCarloOffPolicyAgent(MonteCarloAgent):
    """
    Agente de Monte Carlo off-policy que evalúa y mejora una política objetivo (target)
    mientras se genera la experiencia con una política de comportamiento (behavior).
    
    Se utiliza el muestreo por importancia ponderado para corregir la diferencia entre
    ambas políticas.
    """
    
    def _init_algorithm_params(self, **kwargs):
        """
        Inicializa parámetros específicos para Monte Carlo off-policy.
        Se espera que se reciba la política objetivo mediante 'target_policy'. 
        Si no se especifica, se utilizará la misma que la de comportamiento.
        """
        super()._init_algorithm_params(**kwargs)
        self.target_policy = kwargs.get('target_policy', self.policy)
        # Se puede utilizar la misma tabla de visitas para acumular pesos
        # En este ejemplo, usaremos visit_counts para acumular el total ponderado.
        self.C = np.zeros((self.n_states, self.n_actions))
    
    def _process_episode(self):
        """
        Procesa el episodio completo utilizando Weighted Importance Sampling.
        """
        if not self.episode_buffer:
            return

        G = 0.0
        W = 1.0  # Peso acumulado
        
        # Procesamos el episodio en orden inverso
        for t in range(len(self.episode_buffer) - 1, -1, -1):
            state, action, reward = self.episode_buffer[t]
            G = reward + self.gamma * G
            
            # Acumulamos el total de pesos para este par (s,a)
            self.C[state, action] += W
            
            # Actualizamos Q de manera incremental:
            self.Q[state, action] += (W / self.C[state, action]) * (G - self.Q[state, action])
            
            # La política objetivo es típicamente greedy, por lo que:
            # Si la acción tomada no coincide con la acción greedy según Q, detenemos el proceso.
            greedy_action = np.argmax(self.Q[state])
            if action != greedy_action:
                break
            
            # Calculamos la probabilidad de la acción según la política de comportamiento
            behavior_probs = self.policy.get_action_probabilities(state, self.Q)
            # Evitamos división por cero
            if behavior_probs[action] == 0:
                break
            W = W / behavior_probs[action]