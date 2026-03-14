import matplotlib.pyplot as plt
import numpy as np
import os

class Plotter:
    def __init__(self, save_dir='./plots'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
    
    def plot_rewards(self, rewards, title='Training Rewards', filename='rewards.png'):
        """绘制奖励曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.title(title)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()
    
    def plot_state(self, states, title='CartPole States', filename='states.png'):
        """绘制 CartPole 状态变化"""
        plt.figure(figsize=(12, 8))
        state_names = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity']
        
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.plot([s[i] for s in states])
            plt.title(state_names[i])
            plt.xlabel('Step')
            plt.ylabel('Value')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()
    
    def plot_q_values(self, q_values, title='Q Values', filename='q_values.png'):
        """绘制 Q 值变化"""
        plt.figure(figsize=(10, 6))
        for i in range(q_values.shape[1]):
            plt.plot(q_values[:, i], label=f'Action {i}')
        plt.title(title)
        plt.xlabel('Step')
        plt.ylabel('Q Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()
    
    def plot_epsilon(self, epsilon_history, title='Epsilon Decay', filename='epsilon.png'):
        """绘制 epsilon 衰减曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(epsilon_history)
        plt.title(title)
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()

if __name__ == '__main__':
    # 示例用法
    plotter = Plotter()
    
    # 生成示例数据
    episodes = 100
    rewards = np.random.randn(episodes).cumsum() + 10
    
    steps = 200
    states = np.random.randn(steps, 4)
    
    q_values = np.random.randn(steps, 2)
    
    epsilon_history = np.linspace(1.0, 0.01, episodes)
    
    # 绘制并保存图表
    plotter.plot_rewards(rewards)
    plotter.plot_state(states)
    plotter.plot_q_values(q_values)
    plotter.plot_epsilon(epsilon_history)
    
    print("Plots saved to ./plots directory")
