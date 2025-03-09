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
import random
import numpy as np

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

class DeepQAgent(Agent):
    """
    Agente basado en Deep Q-Learning para aproximar la función Q mediante redes neuronales.
    Incluye un replay buffer y una target network para estabilizar el entrenamiento.
    """
    def _init_algorithm_params(self, **kwargs):
        # Parámetros del DQN
        self.lr = kwargs.get('lr', 0.001)
        self.batch_size = kwargs.get('batch_size', 32)
        self.replay_buffer_size = kwargs.get('replay_buffer_size', 10000)
        self.target_update_freq = kwargs.get('target_update_freq', 100)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Dimensiones del estado y número de acciones (asumimos que el estado es un vector)
        self.input_dim = self.observation_space.shape[0]
        self.n_actions = self.action_space.n
        
        # Redes Q: actual y target
        self.q_network = DQNNetwork(self.input_dim, self.n_actions).to(self.device)
        self.target_network = DQNNetwork(self.input_dim, self.n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizador
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        
        # Replay buffer: lista de transiciones (state, action, reward, next_state, done)
        self.replay_buffer = []
        self.update_counter = 0

    def get_action_values(self, state):
        """
        Devuelve los valores Q para todas las acciones dado un estado,
        utilizando la red Q actual.
        """
        self.q_network.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
        return q_values.cpu().numpy()[0]
    
    def update(self, state, action, next_state, reward, done, info=None):
        """
        Actualiza la red Q utilizando transiciones almacenadas en el replay buffer.
        Se almacena la transición actual y, si hay suficientes muestras, se realiza
        un paso de optimización.
        """
        # Almacenar la transición en el replay buffer
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.pop(0)
        
        # No se actualiza si el batch es menor al tamaño mínimo
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Seleccionar un batch aleatorio de transiciones
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Predicción Q para los estados actuales
        q_values = self.q_network(states).gather(1, actions)
        
        # Predicción Q para los siguientes estados utilizando la target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values, _ = torch.max(next_q_values, dim=1, keepdim=True)
            target = rewards + self.gamma * max_next_q_values * (1 - dones)
        
        # Cálculo de la pérdida (error cuadrático medio)
        loss = nn.MSELoss()(q_values, target)
        
        # Optimización
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Actualizar la target network de forma periódica
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())