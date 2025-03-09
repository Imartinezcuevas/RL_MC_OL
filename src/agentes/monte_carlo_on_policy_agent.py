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
 
class MonteCarloOnPolicyAgent(MonteCarloAgent):
     """
     Agente de Monte Carlo on-policy (misma política para comportamiento y actualización)
     """
     
     def _process_episode(self):
         """
         Procesa el episodio completo según el algoritmo de Monte Carlo on-policy
         """
         if not self.episode_buffer:
             return
             
         # Calcula los retornos (G) para cada paso del episodio
         G = 0
         returns = []
         
         # Recorre el episodio en orden inverso para calcular los retornos
         for t in range(len(self.episode_buffer) - 1, -1, -1):
             state, action, reward = self.episode_buffer[t]
             G = reward + self.gamma * G
             returns.insert(0, G)
         
         # Para first-visit Monte Carlo, lleva registro de las visitas
         if self.first_visit:
             visited = set()
         
         # Actualiza los valores Q basados en los retornos
         for t, ((state, action, _), G) in enumerate(zip(self.episode_buffer, returns)):
             # Para first-visit Monte Carlo, solo actualiza si es la primera visita al par estado-acción
             if self.first_visit:
                 if (state, action) in visited:
                     continue
                 visited.add((state, action))
             
             # Actualización incremental
             self.visit_counts[state, action] += 1
             alpha = 1.0 / self.visit_counts[state, action]
             self.Q[state, action] += alpha * (G - self.Q[state, action])