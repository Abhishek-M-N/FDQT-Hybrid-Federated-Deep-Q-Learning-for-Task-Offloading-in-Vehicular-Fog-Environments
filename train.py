import torch
import torch.optim as optim
from fl_server import FLServer
from vfcenv import VFCEnv
from model import RLModel  

num_clients = 5
env = VFCEnv(num_vehicles=num_clients)
global_model = RLModel()
server = FLServer(model=global_model, num_clients=num_clients)

local_models = [RLModel() for _ in range(num_clients)]
optimizers = [optim.Adam(local_models[i].parameters(), lr=0.001) for i in range(num_clients)]

for round in range(10):  # 10 FL rounds
    print(f"\n🔄 Federated Learning Round {round+1}")

    client_weights = []

    for i in range(num_clients):
        state = env.reset()[i]
        total_reward = 0  

        for step in range(50):  
            state_tensor = torch.tensor([state["cpu_load"], state["task_queue"]], dtype=torch.float32).unsqueeze(0)
            q_values = local_models[i](state_tensor)  
            action = torch.argmax(q_values).item()  

            actions = [0] * num_clients  
            actions[i] = action  
            next_states, rewards = env.step(actions)  

            next_state = next_states[i]  
            reward = torch.tensor(rewards[i], dtype=torch.float32, requires_grad=True)  

            loss = -q_values[0, action] * reward  

            total_reward += reward.item()  

            optimizers[i].zero_grad()
            loss.backward()  
            optimizers[i].step()

        print(f"🚗 Client {i+1}: Total Reward = {total_reward}")

        client_weights.append(local_models[i].state_dict())  # ✅ Store state_dict, not model

    # ✅ Ensure we get a dictionary of model weights
    global_weights = server.aggregate_models(client_weights).state_dict()  

    # ✅ Load the correct weights into each local model
    for i in range(num_clients):
        local_models[i].load_state_dict(global_weights)
