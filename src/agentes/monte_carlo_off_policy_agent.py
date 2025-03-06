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

class MonteCarloOffPolicyAgent(MonteCarloAgent):
    """
    Agente de Monte Carlo Off-Policy usando Importance Sampling.
    """
    
    def _process_episode(self):
        """
        Procesa el episodio completo según el algoritmo de Monte Carlo Off-Policy.
        """
        if not self.episode_buffer:
            return
        
        G = 0  # Inicializamos el retorno
        W = 1.0  # Peso de Importance Sampling
        returns = []
        
        # Para first-visit Monte Carlo, llevamos un registro de visitas
        visited = set()  

        for t in range(len(self.episode_buffer) - 1, -1, -1):
            state, action, reward = self.episode_buffer[t]
            G = reward + self.gamma * G
            returns.insert(0, G)

            if self.first_visit:
                if (state, action) in visited:
                    continue
                visited.add((state, action))
            
            # Debug para revisar `G` y `W`
            #print(f"\nEpisodio {t}: Estado={state}, Acción={action}, Recompensa={reward}, G={G:.4f}, W={W:.4f}")

            # Verificar si la probabilidad de la acción es extremadamente pequeña
            behavior_probs = self.policy.get_action_probabilities(state, self.Q)
            if behavior_probs[action] == 0:
                break

            # Evitar que W explote
            W = min(W / behavior_probs[action], 10.0)

            # Evitar valores NaN o Inf en `W`
            if W == 0 or not np.isfinite(W):
                break

            # Actualización con Importance Sampling
            self.visit_counts[state, action] += 1
            alpha = 1.0 / self.visit_counts[state, action]
            self.Q[state, action] += W * alpha * (G - self.Q[state, action])