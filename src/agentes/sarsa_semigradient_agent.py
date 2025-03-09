"""
Module: agentes/linear_approximate_agent.py
Description: Implementación del algoritmo SARSA com aproximacion semi-gradiente.

Author: Iván Martínez Cuevas
Email: ivan.martinezc@um.es
Date: 2025/02/28

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from agentes.agent import Agent
import torch.nn as nn
import torch.optim as optim
import torch

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(DQNNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class SARSASemiGradientAgent(Agent):
    """
    Agente basado en SARSA semi-gradiente para aproximar la función Q mediante una red neuronal.
    Actualiza la red de forma on-policy utilizando la política epsilon-greedy.
    """
    def _init_algorithm_params(self, **kwargs):
        self.lr = kwargs.get('lr', 0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = self.observation_space.shape[0]
        self.n_actions = self.action_space.n
        
        # Red neuronal para aproximar Q(s,a)
        self.q_network = DQNNetwork(self.input_dim, self.n_actions).to(self.device)
        
        # Optimizador
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        
    def get_action_values(self, state):
        """
        Devuelve los valores Q para todas las acciones dado un estado,
        utilizando la red Q.
        """
        self.q_network.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
        return q_values.cpu().numpy()[0]
    
    def update(self, state, action, next_state, reward, done, info=None):
        """
        Actualiza la red Q usando la regla de SARSA semi-gradiente:
        
            Q(s,a) ← Q(s,a) + lr * [r + γ * Q(s',a';θ) - Q(s,a;θ)]
            
        donde a' se selecciona utilizando la política epsilon-greedy.
        """
        self.q_network.train()
        
        # Convertir estados a tensores
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Valor Q para el estado actual
        q_values = self.q_network(state_tensor)
        current_q = q_values[0, action]
        
        # Calcular el target
        if done:
            target = torch.tensor(reward, dtype=torch.float32).to(self.device)
        else:
            # Seleccionar la siguiente acción con la política (epsilon-greedy)
            next_q_values = self.get_action_values(next_state)
            next_action = self.policy.select_action(next_state, next_q_values)
            self.q_network.eval()
            with torch.no_grad():
                q_next = self.q_network(next_state_tensor)[0, next_action]
            target = torch.tensor(reward, dtype=torch.float32).to(self.device) + self.gamma * q_next
        
        # Calcular el error TD
        td_error = target - current_q
        
        # Pérdida: error cuadrático
        loss = td_error.pow(2)
        
        # Actualizar la red mediante gradiente descendiente
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()